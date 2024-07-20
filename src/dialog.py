# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Dialogue runner class. Implementes communication between two Agents.
"""
import sys
import pdb
import logging
import numpy as np

from metric import MetricsContainer
import data
import utils
import domain

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(filename)s : %(message)s', level=logging.INFO)

class DialogLogger(object):
    """Logger for a dialogue."""
    CODE2ITEM = [
        ('item0', 'book'),
        ('item1', 'hat'),
        ('item2', 'ball'),
    ]
    # CODE2ITEM = [
    #     ('item0', 'Firewood'),
    #     ('item1', 'Food'),
    #     ('item2', 'Water'),
    # ]

    def __init__(self, verbose=False, log_file=None, append=False):
        self.logs = []
        if verbose:
            self.logs.append(sys.stderr)
        if log_file:
            flags = 'a' if append else 'w'
            self.logs.append(open(log_file, flags))

    def _dump(self, s, forced=False):
        for log in self.logs:
            print(s, file=log)
            log.flush()
        if forced:
            print(s, file=sys.stdout)
            sys.stdout.flush()

    def _dump_with_name(self, name, s):
        self._dump('{0: <5} : {1}'.format(name, s))

    def dump_ctx(self, name, ctx):
        assert len(ctx) == 6, 'we expect 3 objects'
        s = ' '.join(['%s=(count:%s value:%s)' % (self.CODE2ITEM[i][1], ctx[2 * i], ctx[2 * i + 1]) \
            for i in range(3)])
        self._dump_with_name(name, s)

    def dump_sent(self, name, sent):
        self._dump_with_name(name, ' '.join(sent))

    def dump_choice(self, name, choice):
        def rep(w):
            p = w.split('=')
            if len(p) == 2:
                for k, v in self.CODE2ITEM:
                    if p[0] == k:
                        return '%s=%s' % (v, p[1])
            return w

        self._dump_with_name(name, ' '.join([rep(c) for c in choice]))

    def dump_agreement(self, agree):
        self._dump('Agreement!' if agree else 'Disagreement?!')

    def dump_reward(self, name, agree, reward):
        if agree:
            self._dump_with_name(name, '%d points' % reward)
        else:
            self._dump_with_name(name, '0 (potential %d)' % reward)

    def dump(self, s, forced=False):
        self._dump(s, forced=forced)


class DialogSelfTrainLogger(DialogLogger):
    """This logger is used to produce new training data from selfplaying."""
    def __init__(self, verbose=False, log_file=None):
        super(DialogSelfTrainLogger, self).__init__(verbose, log_file)
        self.name2example = {}
        self.name2choice = {}

    def _dump_with_name(self, name, sent):
        for n in self.name2example:
            if n == name:
                self.name2example[n] += " YOU: "
            else:
                self.name2example[n] += " THEM: "

            self.name2example[n] += sent

    def dump_ctx(self, name, ctx):
        self.name2example[name] = ' '.join(ctx)

    def dump_choice(self, name, choice):
        self.name2choice[name] = ' '.join(choice)

    def dump_agreement(self, agree):
        if agree:
            for name in self.name2example:
                for other_name in self.name2example:
                    if name != other_name:
                        self.name2example[name] += ' ' + self.name2choice[name]
                        self.name2example[name] += ' ' + self.name2choice[other_name]
                        self._dump(self.name2example[name])

    def dump_reward(self, name, agree, reward):
        pass


class Dialog(object):
    """Dialogue runner."""
    def __init__(self, agents, args, scale_rw = 1.0, rw_type="own_points", conf=None):
        # for now we only suppport dialog of 2 agents
        assert len(agents) == 2
        self.agents = agents
        self.args = args
        self.domain = domain.get_domain(args.domain)
        self.metrics = MetricsContainer()
        self._register_metrics()
        self.scale_rw = scale_rw
        self.rw_type = rw_type
        self.conf = conf

    def _register_metrics(self):
        """Registers valuable metrics."""
        self.metrics.register_average('dialog_len')
        self.metrics.register_average('sent_len')
        self.metrics.register_percentage('agree')
        self.metrics.register_average('advantage')
        self.metrics.register_time('time')
        self.metrics.register_average('comb_rew')
        for agent in self.agents:
            self.metrics.register_average('%s_rew' % agent.name)
            self.metrics.register_percentage('%s_sel' % agent.name)
            self.metrics.register_uniqueness('%s_unique' % agent.name)
        # text metrics
        ref_text = ' '.join(data.read_lines(self.args.ref_text))
        self.metrics.register_ngram('full_match', text=ref_text)

    def _is_selection(self, out):
        return len(out) == 1 and out[0] == '<selection>'

    def show_metrics(self):
        return ' '.join(['%s=%s' % (k, v) for k, v in self.metrics.dict().items()])

    def run(self, ctxs, logger):
        """Runs one instance of the dialogue."""
        assert len(self.agents) == len(ctxs)

        #obj for storage
        storage = {
            "ctxs": {},
            "conv": [],
            "choices": {},
            "agreement_status": None,
            "rewards": {},
        }

        # initialize agents by feeding in the contexes
        for agent, ctx in zip(self.agents, ctxs):
            agent.feed_context(ctx)
            logger.dump_ctx(agent.name, ctx)
            storage["ctxs"][agent.name] = ctx
        logger.dump('-' * 80)

        # choose who goes first by random
        if np.random.rand() < 0.5:
            writer, reader = self.agents
        else:
            reader, writer = self.agents

        conv = []
        # reset metrics
        self.metrics.reset()

        max_utts = 20
        curr = 0
        while curr < max_utts:
            # produce an utterance
            out = writer.write()

            self.metrics.record('sent_len', len(out))
            self.metrics.record('full_match', out)
            self.metrics.record('%s_unique' % writer.name, out)

            # append the utterance to the conversation
            conv.append(out)
            storage["conv"].append(
                {
                    "name": writer.name,
                    "sent": " ".join(out),
                }
            )
            # make the other agent to read it
            reader.read(out)
            if not writer.human:
                logger.dump_sent(writer.name, out)
            # check if the end of the conversation was generated
            if self._is_selection(out):
                self.metrics.record('%s_sel' % writer.name, 1)
                self.metrics.record('%s_sel' % reader.name, 0)
                break
            writer, reader = reader, writer

            curr += 1

        choices = []
        if not self._is_selection(conv[-1]):
            # the conversation did not finish; assume disagreement.
            assert curr == max_utts, curr
            agree, rewards = False, [0 for _ in range(len(ctxs))]

            choices = [
                ["<no_agreement>", "<no_agreement>", "<no_agreement>"],
                ["<no_agreement>", "<no_agreement>", "<no_agreement>"],
            ]

            storage["agreement_status"] = "no_agreement_len"

            for agent, choice, in zip(self.agents, choices):
                storage["choices"][agent.name] = choice

        else:
            # the conversation atleast finished nicely; now we try to get a consistent output.
            # generate choices for each of the agents
            for agent in self.agents:
                choice = agent.choose()
                choices.append(choice)
                logger.dump_choice(agent.name, choice[: self.domain.selection_length() // 2])
                storage["choices"][agent.name] = choice[: self.domain.selection_length() // 2]

            # evaluate the choices, produce agreement and a reward
            agree, rewards = self.domain.score_choices(choices, ctxs, rw_type=self.rw_type, conf=self.conf)
        
        print(agree, rewards)
        if agree == -1 and rewards == -1:
            # this is neither an agreement, nor a disagreement - we don't know due to model failure.
            # print("Failure mode. - agree and rewards are both None. Ignoring this case.")
            print("Failure")

            storage["agreement_status"] = "mismatch_failure" # the choices of the two agents were different, hence, the output is inconclusive.
            return None, None, None, storage
        
        if not agree:
            # this is disagreement between the two.
            # print("Disagreement between the two models.")
            print("Disagreement")
            if not storage["agreement_status"]:
                # there is no agreement, which is not of type len.hence, it is of type wa.
                storage["agreement_status"] = "no_agreement_wa" #the choices match and end in a disagreement.
        else:
            # there is agreement
            print("Agreement")
            storage["agreement_status"] = "agreement" # choices match and are numbers.
        
        for agent, reward in zip(self.agents, rewards):
            if storage["agreement_status"] == "agreement":
                storage["rewards"][agent.name] = reward
            elif "no_agreement" in storage["agreement_status"]:
                storage["rewards"][agent.name] = 0

        logger.dump('-' * 80)

        logger.dump_agreement(agree)
        # perform update, in case if any of the agents is learnable
        for agent, reward in zip(self.agents, rewards):
            logger.dump_reward(agent.name, agree, reward)
            logging.debug("%s : %s : %s" % (str(agent.name), str(agree), str(rewards)))
            agent.update(agree, reward, scale_rw = self.scale_rw)

        if agree:
            self.metrics.record('advantage', rewards[0] - rewards[1])
        self.metrics.record('time')
        self.metrics.record('dialog_len', len(conv))
        self.metrics.record('agree', int(agree))
        self.metrics.record('comb_rew', np.sum(rewards) if agree else 0)
        for agent, reward in zip(self.agents, rewards):
            self.metrics.record('%s_rew' % agent.name, reward if agree else 0)

        logger.dump('-' * 80)
        logger.dump(self.show_metrics())
        logger.dump('-' * 80)
        for ctx, choice in zip(ctxs, choices):
            logger.dump('debug: %s %s' % (' '.join(ctx), ' '.join(choice)))

        return conv, agree, rewards, storage
