# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Reinforcement learning via Policy Gradient (REINFORCE).

Perform RL in bulk on different reward definitions.

Further, perform this reinforce RL in a general manner - against an ensemble of opponents.

New hyperparameters

- nepoch -> nepoch_per_opp: How long to train with a current opponent?
- num_opp_used: How many times should we use an opponent to train with?
- alice_model_file -> policy_model: The model that is being trained.
- bob_model_file -> opp_models: The models that are being trained against.
"""

import argparse
import pdb
import random
import re
import time
import logging
import numpy as np
import torch
from torch import optim
from torch import autograd
import torch.nn as nn
import copy

import config
import data
import utils
from engine import Engine
from utils import ContextGenerator
from agent import LstmAgent, LstmRolloutAgent, RlAgent
from dialog import Dialog, DialogLogger

logging.basicConfig(format=config.log_format, level=config.log_level)

# global counter tracking the number of iterations that have successfully happened.
global_counter = 0

class Reinforce(object):
    """Facilitates a dialogue between two agents and constantly updates them."""
    def __init__(self, dialog, ctx_gen, args, engine, corpus, logger=None):
        self.dialog = dialog
        self.ctx_gen = ctx_gen
        self.args = args
        self.engine = engine
        self.corpus = corpus
        self.logger = logger if logger else DialogLogger()

    def run(self):
        """Entry point of the training."""
        global global_counter

        assert global_counter == 0

        validset, validset_stats = self.corpus.valid_dataset(self.args.bsz,
            device_id=self.engine.device_id)
        trainset, trainset_stats = self.corpus.train_dataset(self.args.bsz,
            device_id=self.engine.device_id)
        N = len(self.corpus.word_dict)

        n = 0
        for ctxs in self.ctx_gen.iter(self.args.nepoch_per_opp):
            n += 1
            if not n % 10:
                logging.info(f"Num: {n}")
            # supervised update
            if self.args.sv_train_freq > 0 and n % self.args.sv_train_freq == 0:
                self.engine.train_single(N, trainset)

            self.logger.dump('=' * 80)
            # run dialogue, it is responsible for reinforcing the agents
            
            self.dialog.run(ctxs, self.logger)
                
            self.logger.dump('=' * 80)
            self.logger.dump('')
            if n % 100 == 0:
                self.logger.dump('%d: %s' % (n, self.dialog.show_metrics()), forced=True)
                logging.info('%d: %s' % (n, self.dialog.show_metrics()))

            # reached here, meaning success; increment global counter.
            global_counter += 1

        def dump_stats(dataset, stats, name):
            loss, select_loss = self.engine.valid_pass(N, dataset, stats)
            self.logger.dump('final: %s_loss %.3f %s_ppl %.3f' % (
                name, float(loss), name, np.exp(float(loss))),
                forced=True)
            self.logger.dump('final: %s_select_loss %.3f %s_select_ppl %.3f' % (
                name, float(select_loss), name, np.exp(float(select_loss))),
                forced=True)

        dump_stats(trainset, trainset_stats, 'train')
        dump_stats(validset, validset_stats, 'valid')

        self.logger.dump('final: %s' % self.dialog.show_metrics(), forced=True)


def main():
    global global_counter

    parser = argparse.ArgumentParser(description='Reinforce')
    parser.add_argument('--data', type=str, default=config.data_dir,
        help='location of the data corpus')
    parser.add_argument('--unk_threshold', type=int, default=config.unk_threshold,
        help='minimum word frequency to be in dictionary')
    # parser.add_argument('--alice_model_file', type=str,
    #     help='Alice model file')
    # parser.add_argument('--bob_model_file', type=str,
    #     help='Bob model file')
    parser.add_argument('--output_model_file', type=str,
        help='output model file')
    parser.add_argument('--context_file', type=str,
        help='context file')
    parser.add_argument('--temperature', type=float, default=config.rl_temperature,
        help='temperature')
    parser.add_argument('--cuda', action='store_true', default=config.cuda,
        help='use CUDA')
    parser.add_argument('--verbose', action='store_true', default=config.verbose,
        help='print out converations')
    parser.add_argument('--seed', type=int, default=config.seed,
        help='random seed')
    parser.add_argument('--score_threshold', type=int, default=config.rl_score_threshold,
        help='successful dialog should have more than score_threshold in score')
    parser.add_argument('--log_file', type=str, default='',
        help='log successful dialogs to file for training')
    parser.add_argument('--smart_bob', action='store_true', default=False,
        help='make Bob smart again')
    parser.add_argument('--gamma', type=float, default=config.rl_gamma,
        help='discount factor')
    parser.add_argument('--eps', type=float, default=config.rl_eps,
        help='eps greedy')
    parser.add_argument('--nesterov', action='store_true', default=config.nesterov,
        help='enable nesterov momentum')
    parser.add_argument('--momentum', type=float, default=config.rl_momentum,
        help='momentum for sgd')
    parser.add_argument('--lr', type=float, default=config.rl_lr,
        help='learning rate')
    parser.add_argument('--clip', type=float, default=config.rl_clip,
        help='gradient clip')
    parser.add_argument('--rl_lr', type=float, default=config.rl_reinforcement_lr,
        help='RL learning rate')
    parser.add_argument('--rl_clip', type=float, default=config.rl_reinforcement_clip,
        help='RL gradient clip')
    parser.add_argument('--ref_text', type=str,
        help='file with the reference text')
    parser.add_argument('--bsz', type=int, default=config.rl_bsz,
        help='batch size')
    parser.add_argument('--sv_train_freq', type=int, default=config.rl_sv_train_freq,
        help='supervision train frequency')
    # parser.add_argument('--nepoch', type=int, default=config.rl_nepoch,
    #     help='number of epochs')
    parser.add_argument('--visual', action='store_true', default=config.plot_graphs,
        help='plot graphs')
    parser.add_argument('--domain', type=str, default=config.domain,
        help='domain for the dialogue')
    
    # kc args
    parser.add_argument('--scale_rw', type=float, default=config.scale_rw,
        help='Scale RL reward')
    parser.add_argument('--rw_type', type=str, default=config.rw_type,
        help='reward type for the reinforce model.')
    
    #kc args - define the new hyperparameters for reinforce generic
    parser.add_argument('--nepoch_per_opp', type=int, default=config.rl_nepoch_per_opp,
        help='number of epochs per opponent')
    parser.add_argument('--num_opp_used', type=int, default=config.num_opp_used,
        help='How many times should we use an opponent to train with?')
    parser.add_argument('--policy_model', type=str,
        help='The model that is being trained.')
    parser.add_argument('--opp_models', type=str,
        help='The list of models that are being trained against.')
    
    args = parser.parse_args()
    args.opp_models = args.opp_models.split(',')
    print(args)

    device_id = utils.use_cuda(args.cuda)
    logging.info("Starting training using pytorch version:%s" % (str(torch.__version__)))
    logging.info("CUDA is %s" % ("enabled. Using device_id:"+str(device_id) + " version:" \
        +str(torch.version.cuda) + " on gpu:" + torch.cuda.get_device_name(0) if args.cuda else "disabled"))

    # let's only support this for when the rw_type is utility
    assert args.rw_type == "utility", "Only utility reward type is supported for now."

    # train multiple models based on configurations in config.utilities
    print(f"Total configurations: {len(config.utility_configs)}")

    for conf_index in range(len(config.utility_configs)):
        conf = config.utility_configs[conf_index]

        logging.info(f"Using config {conf_index}: {' '.join([str(ix) for ix in conf])}")

        alice_model = utils.load_model(args.policy_model)
        # we don't want to use Dropout during RL
        alice_model.eval()

        prev_alice_model = copy.deepcopy(alice_model)
        
        opp_index = 0
        
        for opp_used_no in range(args.num_opp_used):

            # we choose the next opponent model
            logging.info(f"Use opponent {opp_used_no + 1} of {args.num_opp_used}")
            chosen_opp_model = args.opp_models[opp_index]

            while True:
        
                # Alice is a RL based agent, meaning that she will be learning while selfplaying
                logging.info("Creating RlAgent from policy_model: %s" % (args.policy_model))
                alice_model = copy.deepcopy(prev_alice_model)
                alice = RlAgent(alice_model, args, name='Alice')
            
                # we keep Bob frozen, i.e. we don't update his parameters
                logging.info("Creating Bob's (--smart_bob) LstmRolloutAgent" if args.smart_bob \
                    else "Creating Bob's (not --smart_bob) LstmAgent" )
                logging.info("Creating Bob from chosen_opp_model: %s" % (chosen_opp_model))
                bob_ty = LstmRolloutAgent if args.smart_bob else LstmAgent
                bob_model = utils.load_model(chosen_opp_model)
                bob_model.eval()
                bob = bob_ty(bob_model, args, name='Bob')

                logging.info("Initializing communication dialogue between Alice and Bob")
                dialog = Dialog([alice, bob], args, scale_rw=args.scale_rw, rw_type=args.rw_type, conf=conf)
                logger = DialogLogger(verbose=args.verbose, log_file=args.log_file)
                ctx_gen = ContextGenerator(args.context_file)

                logging.info("Building word corpus, requiring minimum word frequency of %d for dictionary" % (args.unk_threshold))
                corpus = data.WordCorpus(args.data, freq_cutoff=args.unk_threshold)
                engine = Engine(alice_model, args, device_id, verbose=False)

                logging.info("Starting Reinforcement Learning")
                reinforce = Reinforce(dialog, ctx_gen, args, engine, corpus, logger)
                try:
                    reinforce.run()
                except RuntimeError:
                    print("runtime error caught !!!")

                if global_counter >= args.nepoch_per_opp:
                    # atleast some significant iterations happened properly
                    
                    # reset the global counter
                    global_counter = 0

                    # update the prev model
                    prev_alice_model = copy.deepcopy(alice.model)

                    #now break out of the loop to continue to the next opponent.
                    break

                else:
                    # redo the current opp; we could not get enough iterations with the current opponent.

                    # reset the global counter
                    global_counter = 0

            # out of the while loop; so we are done with this opponent.
            opp_index += 1
            opp_index = opp_index % len(args.opp_models)

        out_path = f"{args.output_model_file.replace('.pt', '')}_generic_rw_{args.rw_type}_{'_'.join([str(ix) for ix in conf])}.pt"
        logging.info("Saving updated Alice model to %s" % (out_path))
        utils.save_model(prev_alice_model, out_path)


if __name__ == '__main__':
    main()
