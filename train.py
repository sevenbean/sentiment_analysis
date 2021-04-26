from Utils.dataUtil import Tokenizer,build_tokenizer
from Utils.ReviewDataset import ReviewDataSet
from torch.utils.data import DataLoader
from Mymodel.lstm import  LSTM
import argparse
import torch
from torch.utils.data import random_split
import  logging
import sys
import os
from sklearn import  metrics
logger=logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
class Instruction(object):
    def __init__(self,opt):
        super(Instruction, self).__init__()
        self.opt=opt
        self.tokenizer = build_tokenizer(opt.max_seq_len, opt.wordvec_path, opt.embed_dim, opt.dataset_file)
        self.reDataset = ReviewDataSet(self.tokenizer, opt.dataset_file)
        if (opt.testset_radio > 0):
            testset_radio = int(len(self.reDataset) * opt.testset_radio)
            self.reDataset_train, self.reDataset_test = random_split(self.reDataset, (len(self.reDataset) - testset_radio, testset_radio))
        self.embedding_matrix=self.tokenizer.build_embeding_matrix()
        self.model = LSTM(self.embedding_matrix, self.opt).to(self.opt.device)

    def _evaluate_acc_f1(self,data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch Mymodel to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = t_sample_batched["text"].to(self.opt.device)
                t_targets = t_sample_batched['label'].to(self.opt.device)
                t_outputs = self.model(t_inputs)
                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(),pos_label=1)
        return acc, f1

    def _train(self,reDataloader_train,reDataloader_test,optimizer,criterion):
        max_test_acc = 0
        max_test_f1 = 0
        global_step = 0
        for epoch in range(self.opt.num_epoch):
            n_correct, n_total, loss_total = 0, 0, 0
            self.model.train()
            for t_batchsize, inputs in enumerate(reDataloader_train):
                global_step += 1
                optimizer.zero_grad()
                outputs = self.model(inputs["text"].to(self.opt.device))
                targets = inputs["label"].to(self.opt.device)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                # 正确的数量
                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                # 总数
                n_total += len(outputs)
                loss_total = loss.item() * len(outputs)
                if global_step % 5 == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))
                    # 验证集评估
            val_acc, val_f1 = self._evaluate_acc_f1(reDataloader_test)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))
            if val_acc > max_test_acc:
                max_test_acc = val_acc
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                # 保存model
                torch.save(self.model.state_dict(), "state_dict/best_model")
            if val_f1 > max_test_f1:
                max_test_f1 = val_f1


    def _run(self):

        reDataloader_train = DataLoader(self.reDataset_train, batch_size=self.opt.batchsize, shuffle=True)
        reDataloader_test = DataLoader(self.reDataset_test, batch_size=self.opt.batchsize, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.02)
        criterion = torch.nn.CrossEntropyLoss()
        self._train(reDataloader_train,reDataloader_test,optimizer,criterion)

    def _predict(self,text):
        self.model.load_state_dict(torch.load("state_dict/best_model"))
        self.model.eval()
        sequence=self.tokenizer.text_to_sequence(text)
        with torch.no_grad():
            sequences=torch.tensor(sequence).unsqueeze(dim=0).to(torch.cuda.current_device())
            outputs=self.model(sequences)
            targets=torch.argmax(outputs,dim=-1)
            polarities = {0: "消极", 1: "积极"}

            print("{}  情感极性：{}".format(text,polarities[targets.cpu().numpy()[0]]))



def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("--max_seq_len",default=100,type=int)
    arg.add_argument("--embed_dim",default=300,type=int)
    arg.add_argument("--wordvec_path",default="./wordvec/sgns.zhihu.bigram.bz2",type=str)
    arg.add_argument("--batchsize",default=32,type=int)
    arg.add_argument("--device",default=torch.cuda.current_device())
    arg.add_argument("--hidden_dim",default=300,type=int)
    arg.add_argument("--polarities_dim",default=2,type=int)
    arg.add_argument("--num_epoch",default=10,type=int)
    arg.add_argument("--testset_radio",default=0.3,type=int)
    opt=arg.parse_args()
    opt.dataset_file=["./dataset/negative_samples.txt","./dataset/positive_samples.txt"]
    ins=Instruction(opt)
    # ins._run()

    ins._predict("早餐还可以，无论去多少人，那边也不加食品的。酒店应该重视一下这个问题了。")

if __name__ == '__main__':
    main()















