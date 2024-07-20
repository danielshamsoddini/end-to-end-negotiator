import os
import openai
import logging
import time
import argparse
import json

from agent import NegotiationAgent, PartnerAgent, ModeratorAgent
from utils import load_txt_file, convert_item_cnts_partner, calculate_score, compute_time, setup_logging

from .. import dialog
from .. import utils as rl_utils
from ..models import dialog_model as rl_model
 
class RLAgent:
    def __init__(self):
        self.agent = rl_model.LstmRolloutAgent(rl_utils.load_model(args.alice_model_file), args, name = 'RL-Agent')
    
    def respond(self, convo):
        self.agent.read(convo)
        return self.agent.write()
    

    


class AgentMaster:
    def __init__(self, agents):
        self.agents = agents

    def reset_agents(self):
        for agent in self.agents.values():
            agent.reset()

    def finalize_deal(self, offer_history, agent_value_off_table, partner_value_off_table, label, evaluation_metrics):
        final_agreement = offer_history[-1]
        partner_agreement = convert_item_cnts_partner(final_agreement, only_cnts=True)
        agent_final_score = calculate_score(final_agreement, agent_value_off_table)
        partner_final_score = calculate_score(partner_agreement, partner_value_off_table)
        joint_score = agent_final_score + partner_final_score
        logging.info(f"[{label}] Agent's agreement: {final_agreement} ({agent_final_score})")
        logging.info(f"[{label}] Partner's agreement: {partner_agreement} ({partner_final_score})")

        evaluation_metrics["agent_score"]=agent_final_score
        evaluation_metrics["partner_score"]=partner_final_score
        evaluation_metrics["joint_score"]=joint_score
        evaluation_metrics["win"] = 1 if agent_final_score > partner_final_score else 0
        #return agent_final_score, partner_final_score, joint_score

    def conduct_single_simulation(self, agent_1, agent_2, moderator=None, n_round=10, turn_level_verification=False):

        self.reset_agents()
        assert agent_1.agent_type == "NegotiatorAgent", "The first agent should be NegotiatorAgent in the current setting."

        if moderator is None:
            moderator = self.agents['ModeratorAgent']

        # player information
        logging.info("**** Player Information ****")
        logging.info("> Agent1: %s", agent_1.agent_type)
        logging.info("> Agent2: %s", agent_2.agent_type)
        logging.info("> Moderator: %s", moderator.agent_type)

        #Initial Utterance
        logging.info("\n")
        logging.info("**** Starting the conversation ****")
        initial_partner_utterance = "Hello!"
        agent_1_response= agent_1.respond(initial_partner_utterance)
        logging.info(f"> {agent_2.agent_type} Response: {initial_partner_utterance}")
        logging.info("-" *100)
        logging.info(f"> {agent_1.agent_type} Response: {agent_1_response}")
        logging.info("-" *100)

        evaluation_metrics={"agent_score": None, "partner_score": None, "joint_score": None, "num_turns": None, "agreement": None, "win": None}
        for idx in range(n_round):

            #Negotiatiator_response = Negotiatiator.respond(initial_utterance)
            agent_2_response = agent_2.respond(agent_1_response)
            logging.info(f"> {agent_2.agent_type}: {agent_2_response}")
            logging.info("-" *100)
            agent_1_response = agent_1.respond(agent_2_response)
            logging.info(f"> {agent_1.agent_type}: {agent_1_response}")
            logging.info("-" *100)

            if turn_level_verification:
                input("Holding for the turn-level response verifiaction. Press Enter to continue...")


            if dialog._is_selection(agent_2_response):
                choice = agent_2.agent.choose()
                print(choice)
                # evaluate the choices, produce agreement and a reward
                # agree, rewards = self.domain.score_choices(choices, ctxs, rw_type=self.rw_type, conf=self.conf)
                #this code was for 2 rl agents, we need to score choices ourself

            # TODO : Deal Checking for the agent_2_response
            if "ACCEPT-DEAL" in agent_2_response: #agent_2 accepts first to agent_1's offer
               self.finalize_deal(agent_1.offer_history, agent_1.agent_value_off_table, partner_agent_value_off_table, "A2-Accept", evaluation_metrics)

            elif "ACCEPT-DEAL" in agent_1_response: #agent_1 accepts first to agent_2's offer
               self.finalize_deal(agent_1.partner_offer_history, agent_1.agent_value_off_table, partner_agent_value_off_table, "A1-Accept", evaluation_metrics)

            check_status = moderator.check_status(agent_1.dialog_history)
            logging.info(f"( Check Status by Moderator : {check_status} )")
            logging.info("-" *100)

            if check_status in ["ACCEPT-DEAL"]:
                logging.info("**** The negotiation is over : DEAL ****")
                evaluation_metrics['num_turns'] = idx+1
                evaluation_metrics['agreement'] = "AGREEMENT"
                break
            elif check_status in ["WALK-AWAY"]:
                # Double check the status given latest utterance
                if "WALK-AWAY" in agent_1_response or "WALK-AWAY" in agent_2_response:
                    logging.info("**** The negotiation is over : WALK-AWAY ****")
                    evaluation_metrics['num_turns'] = idx+1
                    evaluation_metrics['agreement'] = "WALK-AWAY"
                    break
                else:
                    logging.info("* The moderator checked the status as WALK-AWAY, but WALK-AWAY was not found in the response. Continuing.. ")

            if idx == n_round-1:
                logging.info("**** The negotiation is over : MAX TURNS ****")
                evaluation_metrics['agreement'] = "NO-AGREEMENT"
                break

        return evaluation_metrics

    def run_experiment(self, agent_1, agent_2, n_exp=1, n_round=10, turn_level_verification=False):
        """run multiple experiments
        """
        start_time = time.time()
        output = {"agent_score": [], "partner_score": [], "joint_score": [], "num_turns": [], "agreement": [], 'win': []}
        for i in range(n_exp):
            logging.info("==== ver %s CASE (%d / %d) ====" % ("test", i, n_exp))
            logging.info('Agent1 (%s) vs Agent2 (%s)\n' % (agent_1.agent_type, agent_2.agent_type))

            _output= self.conduct_single_simulation(agent_1, agent_2, n_round=n_round, turn_level_verification=turn_level_verification)
            for key in output.keys():
                output[key].append(_output[key])
            logging.info("agent_score: %s | partner_score: %s | joint_score: %s | num_turns: %s | agreement: %s | win: %s" % (
                _output["agent_score"], _output["partner_score"], _output["joint_score"], _output["num_turns"], _output["agreement"], _output["win"]))
            logging.info("==== End: elapsed time %.2f min ====" % compute_time(start_time))
            logging.info("\n\n")
        return output


def parse_experiment_args(parser):
    parser.add_argument('--n_round', type=int, default=10, help='Number of rounds for the negotiation')
    parser.add_argument('--n_exp', type=int, default=1, help='Number of experiments')
    parser.add_argument('--agent1', type=str, default='NegotiatorAgent', help='Agent1 type')
    parser.add_argument('--agent2', type=str, default='PartnerAgent', help='Agent2 type')
    parser.add_argument('--moderator', type=str, default='ModeratorAgent', help='Moderator type')
    parser.add_argument('--engine', type=str, default='gpt-4o', help='OpenAI Engine')
    parser.add_argument('--api-key', type=str, default='', help='OpenAI API Key')
    parser.add_argument('--partner-agent-personality', type=str, default='greedy', help='Partner Agent Personality')
    parser.add_argument('-tlrv', '--turn-level-response-verification', action='store_true', help='Turn-level response verification')
    return parser

def parse_agent_args(parser):
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')
    parser.add_argument('-iv', '--inconsistency-verbose', action='store_true', help='Verbose Info')
    parser.add_argument('-w1', '--weight-OSAD', type=float, default=0.5, help='the weight of OSAD', )
    parser.add_argument('-w2', '--weight-self-assessment', type=float, default=0.5, help='the weight of self-assessment')
    parser.add_argument('-upc', '--update-partner-priority-in-checker',  type=bool, default=True, help='True of False for updating partner priority in checker')
    parser.add_argument('-on-cc', '--priority-consistency-checker-on',  type=bool, default=True, help='True of False for turning on the consistency checker')
    parser.add_argument('-lp-caching', '--lp-caching-on', action='store_true', help='True of False for turning on caching for LP')
    return parser

if __name__=="__main__":
    # Set up the logging.
    setup_logging(logging.INFO) # For debugging, set the level=logging.DEBUG

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Agent-Agent Simulation')
    parser = parse_experiment_args(parser)
    parser = parse_agent_args(parser)
    args = parser.parse_args()
    logging.info("**** Arguments for Simulation ****")
    logging.info(json.dumps(vars(args), indent=2))
    logging.info("\n")

    args.api_key = os.environ.get("OPENAI_API_KEY") # for safety, set the OPENAI_API_KEY in the environment variable

    #######################
    # Negotiation Setting
    #######################
    # Integrative
    agent_value_off_table = {"food": 5, "water": 4, "firewood": 3}  # agent
    partner_agent_value_off_table = {"food": 3, "water": 4, "firewood": 5}  # agent

    # Distributive
    # agent_value_off_table = {"food": 5, "water": 4, "firewood": 3}  # agent
    # partner_agent_value_off_table = {"food": 5, "water": 4, "firewood": 3}  # agent


    #######################
    # Set up the agent
    #######################
    api_key = args.api_key
    engine = args.engine

    system_instruction = load_txt_file('prompt/system_instruction.txt')
    moderator_instruction = load_txt_file('prompt/moderator_instruction.txt')

    partner_agent_personality = args.partner_agent_personality # "prosocial" "base"
    partner_agent_instruction = load_txt_file(f'prompt/{partner_agent_personality}_partner_instruction.txt') \
                                if partner_agent_personality in ["prosocial", "greedy"] else system_instruction
    partner_agent_type = f"PartnerAgent({partner_agent_personality[0].upper()})" # "RLAgent(G)"
    agents=dict()

    # Mapping the agent type to the agent class
    # To do : Add RL Agents
    agents['NegotiatorAgent'] = NegotiationAgent(agent_value_off_table=agent_value_off_table, system_instruction=system_instruction,
                                agent_type='NegotiatorAgent', engine=engine, api_key=api_key, args=args)
    agents['OSADAgent'] = PartnerAgent(agent_value_off_table=agent_value_off_table, system_instruction=system_instruction,
                                agent_type='OSAD_agent', engine=engine, api_key=api_key)
    agents['ModeratorAgent'] = ModeratorAgent(agent_type='Moderator', engine=engine, api_key= api_key, system_instruction=moderator_instruction,
                                trace_n_history=4, verbose=False)
    agents['PartnerAgent'] = PartnerAgent(agent_value_off_table=partner_agent_value_off_table, system_instruction=partner_agent_instruction,
                                agent_type=partner_agent_type, engine=engine, api_key=api_key)
    agents['NegotiatorAgent'].set_OSAD_agent(agents['OSADAgent'])
    agents['RLAgent'] = RLAgent(agent_value_off_table=agent_value_off_table, system_instruction=system_instruction, agent_type='RLAgent', engine=engine, args = args)



    #######################
    # Agent to Agent Simulation
    #######################
    # set up experiment parameters
    n_exp = args.n_exp
    n_round = args.n_round
    tlrv = args.turn_level_response_verification

    agent1 = agents[args.agent1]
    agent2 = agents[args.agent2]
    moderator = agents[args.moderator]

    agent1.verbose=args.verbose
    agent2.verbose=args.verbose

    Arena= AgentMaster(agents)
    # Arena.conduct_single_simulation(agent_1=agent1,  agent_2=agent2,  n_round=n_round) # Single Simulation
    Arena.run_experiment(agent_1=agent1,  agent_2=agent2, n_exp=n_exp, n_round=n_round, turn_level_verification=tlrv) # Multiple Simulation
