import torch.nn as nn
from torch.optim import lr_scheduler
import time
from datetime import datetime
from utils.utils import *
from model.AD_Model import AD_Model
from data_factory.data_loader import get_loader_segment
from utils.utils import calc_diffusion_hyperparams, std_normal
from model.trend_seasonal import TS_Model
from model.series_decompose import tsr_decomp
from layers.three_sigma_mask import generate_mask_for_timeseries

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))



class EarlyStopping:
    def __init__(self, patience=0, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.train_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, train_loss, val_loss, model, path):
        score = train_loss
        score2 = val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(train_loss, val_loss, model, path)
        elif score2 > self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(train_loss, val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, train_loss, val_loss, model, path):
        if self.verbose:
            print(f'val loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if isinstance(model, AD_Model):
            torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_model_checkpoint.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_premodel_checkpoint.pth'))
        self.train_loss_min = train_loss
        self.val_loss_min = val_loss


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.decompose = tsr_decomp(kernel_size=[20, 25, 30], period=7)


    def build_model(self):
        self.pre_model = TS_Model(seq_len=self.win_size, num_nodes=self.input_c, d_model=128)
        self.model = AD_Model(enc_in=self.input_c, c_out=self.output_c)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.preoptimizer = torch.optim.SGD(self.pre_model.parameters(), lr=self.lr, momentum=0.9)
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.prescheduler = lr_scheduler.CosineAnnealingLR(self.preoptimizer, T_max=100)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model.to(self.device)
            self.pre_model.to(self.device)


    def vali_premodel(self, vali_loader):
        self.pre_model.eval()
        loss2_list = []
        for i, (input_data, _) in enumerate(vali_loader):
            trend, seasonal, residual = self.decompose(input_data)
            normal = trend + seasonal

            input = normal.float().to(self.device)

            output = self.pre_model(input)

            loss = self.criterion(output, input)

            loss2_list.append(loss.item())
        return np.average(loss2_list)


    def vali_model(self, vali_loader):
        self.pre_model.load_state_dict(
            torch.load(os.path.join(str(self.model_save_path), str(self.dataset) + '_premodel_checkpoint.pth')))
        self.model.eval()
        loss2_list = []
        diffusion_hyperparams = calc_diffusion_hyperparams(self.T, self.beta_0, self.beta_T)
        for i, (input_data, _) in enumerate(vali_loader):
            trend, seasonal, residual = self.decompose(input_data)
            normal = trend + seasonal
            normal_update = self.pre_model(normal.to(self.device))
            residual = input_data.to(self.device) - normal_update
            residual = residual.permute(0, 2, 1)
            input = residual.float().to(self.device)

            T, Alpha_bar = diffusion_hyperparams["T"], diffusion_hyperparams["Alpha_bar"].to(self.device)
            series = input
            mask_matrix = generate_mask_for_timeseries(series)
            series = series * mask_matrix
            B, C, L = series.shape
            diffusion_steps = torch.randint(T, size=(B, 1, 1)).to(self.device)
            z = std_normal(series.shape).to(self.device)

            x_t = torch.sqrt(Alpha_bar[diffusion_steps]) * series + torch.sqrt(
                1 - Alpha_bar[diffusion_steps]) * z

            epsilon_theta1, epsilon_theta2 = self.model((x_t, mask_matrix, diffusion_steps.view(B, 1),))

            loss = (1 / 2) * self.criterion(epsilon_theta1, z) + (1 / 2) * self.criterion(epsilon_theta2, z)

            loss2_list.append(loss.item())
        return np.average(loss2_list)




    def train(self):

        print("======================TRAIN MODE======================")

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        pre_early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.dataset)
        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        diffusion_hyperparams = calc_diffusion_hyperparams(self.T, self.beta_0, self.beta_T)


        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = datetime.now()
            self.pre_model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                trend, seasonal, residual = self.decompose(input_data)
                normal = trend + seasonal

                self.preoptimizer.zero_grad()
                iter_count += 1
                input = normal.float().to(self.device)
                output = self.pre_model(input)

                loss = self.criterion(output, input)
                loss1_list.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.preoptimizer.step()
            self.prescheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, datetime.now() - epoch_time))
            train_loss = np.average(loss1_list)
            val_loss = self.vali_premodel(self.test_loader)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, val_loss))
            pre_early_stopping(train_loss, val_loss, self.pre_model, path)
            print('Updating learning rate to {}'.format(self.prescheduler.get_last_lr()))
            if pre_early_stopping.early_stop:
                print("Pre Early stopping")
                break


        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []
            n = epoch + 1

            epoch_time = datetime.now()
            self.pre_model.load_state_dict(
                torch.load(os.path.join(str(self.model_save_path), str(self.dataset) + '_premodel_checkpoint.pth')))
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                trend, seasonal, residual = self.decompose(input_data)
                normal = trend + seasonal
                normal_update = self.pre_model(normal.to(self.device))
                residual = input_data.to(self.device) - normal_update
                residual = residual.permute(0, 2, 1)

                self.optimizer.zero_grad()
                iter_count += 1
                input = residual.float().to(self.device)

                T, Alpha_bar = diffusion_hyperparams["T"], diffusion_hyperparams["Alpha_bar"].to(self.device)
                series = input
                B, C, L = series.shape
                mask_matrix = generate_mask_for_timeseries(series)
                series = series * mask_matrix
                diffusion_steps = torch.randint(T, size=(B, 1, 1)).to(self.device)
                z = std_normal(series.shape).to(self.device)
                x_t = torch.sqrt(Alpha_bar[diffusion_steps]) * series + torch.sqrt(
                    1 - Alpha_bar[diffusion_steps]) * z

                epsilon_theta1, epsilon_theta2 = self.model((x_t, mask_matrix, diffusion_steps.view(B, 1),))

                loss = (1 / n) * self.criterion(epsilon_theta1, z) + (1 - 1/n) * self.criterion(epsilon_theta2, z)
                loss1_list.append(loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, datetime.now() - epoch_time))
            train_loss = np.average(loss1_list)
            val_loss = self.vali_model(self.test_loader)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, val_loss))
            early_stopping(train_loss, val_loss, self.model, path)
            print('Updating learning rate to {}'.format(self.scheduler.get_last_lr()))
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def test(self):
        self.model.load_state_dict(
            torch.load(os.path.join(str(self.model_save_path), str(self.dataset) + '_model_checkpoint.pth')))
        self.pre_model.load_state_dict(
            torch.load(os.path.join(str(self.model_save_path), str(self.dataset) + '_premodel_checkpoint.pth')))
        self.pre_model.eval()
        self.model.eval()

        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduction='none')

        # (1) find the threshold
        start_time1 = datetime.now()
        attens_energy = []
        diffusion_hyperparams = calc_diffusion_hyperparams(self.T, self.beta_0, self.beta_T)
        for i, (input_data, labels) in enumerate(self.thre_loader):
            trend, seasonal, residual = self.decompose(input_data)
            normal = trend + seasonal
            normal_update = self.pre_model(normal.to(self.device))
            residual = input_data.to(self.device) - normal_update
            residual = residual.permute(0, 2, 1)
            input = residual.float().to(self.device)

            T = diffusion_hyperparams["T"]
            Alpha = diffusion_hyperparams["Alpha"]
            Alpha_bar = diffusion_hyperparams["Alpha_bar"]
            Sigma = diffusion_hyperparams["Sigma"].to(self.device)

            size = (input.size(0), input.size(1), input.size(2))
            x = std_normal(size).to(self.device)
            mask_matrix = torch.ones_like(input)
            with torch.no_grad():
                for t in range(T - 1, -1, -1):
                    diffusion_steps = (t * torch.ones((size[0], 1))).to(self.device)
                    epsilon_theta1, epsilon_theta2  = self.model((x, mask_matrix, diffusion_steps,))
                    mean = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * (epsilon_theta1+epsilon_theta2)/2) / torch.sqrt(Alpha[t])
                    if t > 0:
                        x = mean + Sigma[t] * std_normal(size).to(self.device)

            output = x
            loss = torch.mean(criterion(input, output), dim=1)
            cri = loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        thre_energy = np.array(attens_energy)

        thresh = np.percentile(thre_energy, 100 - self.anomaly_ratio)
        end_time1 = datetime.now()
        time1 = end_time1 - start_time1
        print("dataset:", self.dataset)
        print("spend time1:", time1)

        # (2) evaluation on the test set
        sum_f_score = []
        sum_accuracy = []
        sum_precision = []
        sum_recall = []
        for k in range(0, 10):
            test_labels = []
            attens_energy = []
            for i, (input_data, labels) in enumerate(self.thre_loader):
                trend, seasonal, residual = self.decompose(input_data)
                normal = trend + seasonal
                normal_update = self.pre_model(normal.to(self.device))
                residual = input_data.to(self.device) - normal_update
                residual = residual.permute(0, 2, 1)
                input = residual.float().to(self.device)

                T = diffusion_hyperparams["T"]
                Alpha = diffusion_hyperparams["Alpha"]
                Alpha_bar = diffusion_hyperparams["Alpha_bar"]
                Sigma = diffusion_hyperparams["Sigma"].to(self.device)

                size = (input.size(0), input.size(1), input.size(2))
                x = std_normal(size).to(self.device)
                mask_matrix = torch.ones_like(input)
                with torch.no_grad():
                    for t in range(T - 1, -1, -1):
                        diffusion_steps = (t * torch.ones((size[0], 1))).to(self.device)
                        epsilon_theta1, epsilon_theta2 = self.model((x, mask_matrix, diffusion_steps,))
                        mean = (x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * (epsilon_theta1+epsilon_theta2)/2) / torch.sqrt(
                            Alpha[t])
                        if t > 0:
                            x = mean + Sigma[t] * std_normal(size).to(self.device)
                output = x
                loss = torch.mean(criterion(input, output), dim=1)
                cri = loss
                cri = cri.detach().cpu().numpy()
                attens_energy.append(cri)
                test_labels.append(labels)

            attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)

            test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
            test_labels = np.array(test_labels)
            pred = (test_energy > thresh).astype(int)
            gt = test_labels.astype(int)

            # detection adjustment
            anomaly_state = False
            for i in range(len(gt)):
                if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                    anomaly_state = True
                    for j in range(i, 0, -1):
                        if gt[j] == 0:
                            break
                        else:
                            if pred[j] == 0:
                                pred[j] = 1
                    for j in range(i, len(gt)):
                        if gt[j] == 0:
                            break
                        else:
                            if pred[j] == 0:
                                pred[j] = 1
                elif gt[i] == 0:
                    anomaly_state = False
                if anomaly_state:
                    pred[i] = 1

            pred = np.array(pred)
            gt = np.array(gt)
            if k == 0:
                print("pred: ", pred.shape)
                print("gt:   ", gt.shape)

            from sklearn.metrics import precision_recall_fscore_support
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(gt, pred)
            precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                                  average='binary')

            sum_f_score.append(f_score)
            sum_recall.append(recall)
            sum_precision.append(precision)
            sum_accuracy.append(accuracy)
        max_index = sum_f_score.index(np.max(sum_f_score))
        max_f_score = sum_f_score[max_index]
        max_recall = sum_recall[max_index]
        max_precision = sum_precision[max_index]
        max_accuracy = sum_accuracy[max_index]
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                max_accuracy, max_precision,
                max_recall, max_f_score))

