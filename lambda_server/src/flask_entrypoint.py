import random
from flask import Flask
from flask import request
from flask import json
import torch
import utils

app = Flask(__name__)

#globals: to be initialized in initial_setup - a common storage variable for everything, including initiated models and current states of each of the users.
STORAGE = {
  "static": {},
  "users": {}
}
SERVER_STATUS = "Not Ready"


def initial_setup():
  """
  Initial setup for the server
    - Load all the models.
    - Load context pairs and be ready for answering the requests.
  """
  global STORAGE
  global SERVER_STATUS

  # load potential model, context pairs from the json.
  with open("data/negotiate/mod_cxt_pairs.json", "r") as f:
    STORAGE["static"]["mod_cxt_all"] = [tuple(item) for item in json.load(f)["mod_cxt_pairs"]]

  # load all the models - mod is the model object that will be used for reading, writing, etc.
  name2mod = utils.load_models()
  STORAGE["static"]["name2mod"] = name2mod

  # initialize the storage for users
  STORAGE["users"]["mod_cxt_used"] = set() # set of (model, context) pairs that have been used.
  STORAGE["users"]["user_data"] = {} # dict from random IDs to everything related to a specific user, model, cxt, and all lioness interaction.

  # mark the server as ready
  
  SERVER_STATUS = "Ready"


@app.route('/setup_new_user/', methods=['POST'])
def setup_new_user():
  """
  Setup for a new user - build the connection.
   - return a new random id, and human context, for the user.
   - save the model, context and random id in the storage.
   - mark this triplet as used.
  """
  global STORAGE

  if SERVER_STATUS != "Ready":
    # server is not yet ready with the initial setup.
    data = {}
    data["status"] = "Error"
    data["error_description"] = "Server is not ready yet. Please wait for a few seconds and try again."
    return json.dumps(data)

  # pick 10 random digits from 1 to 9 (both included) and join them as the randomId for the user
  randomId = ''.join([str(random.randint(1, 9)) for _ in range(10)])

  # pick a new model, cxt pair that is not in mod_cxt_used
  chosen_mod_cxt = None
  for mod_cxt_pair in STORAGE["static"]["mod_cxt_all"]:
    if mod_cxt_pair not in STORAGE["users"]["mod_cxt_used"]:
      chosen_mod_cxt = mod_cxt_pair
      break
  
  # if no new model, cxt pair is available, pick a pair at random
  if not chosen_mod_cxt:
    # choose any model and a cxt pair at random
    chosen_mod_cxt = random.choice(STORAGE["static"]["mod_cxt_all"])
  
  # update the storage appropriately
  STORAGE["users"]["mod_cxt_used"].add(chosen_mod_cxt)
  STORAGE["users"]["user_data"][randomId] = {
    "model": chosen_mod_cxt[0],
    "cxt": chosen_mod_cxt[1],
    "lioness": {}, # start an account for this user in the storage
  }

  # output
  data = {}
  data["status"] = "Success"
  data["randomId"] = randomId
  data["hct"] = " ".join(chosen_mod_cxt[1].split()[:6])
  data["agct"] = utils.encode(" ".join(chosen_mod_cxt[1].split()[6:]), key="")
  data["agm"] = utils.encode(chosen_mod_cxt[0], key="")
  return json.dumps(data)


@app.route('/model_resp/', methods=['POST'])
def model_resp():
  """
  Get model response - interface for all the cases possible. The method in utils processes the request, and returns the response object (that is sent out), and storage obj (that is used to update the lioness storage for the user).

  Input payload must contain randomId, model, cxt, and any human utterance.

  """
  global STORAGE
  if SERVER_STATUS != "Ready":
    # server is not yet ready with the initial setup.
    data = {}
    data["status"] = "Error"
    data["error_description"] = "Server is not ready yet. Please wait for a few seconds and try again."
    return json.dumps(data)

  payload = json.loads(request.get_data().decode('utf-8'))

  # check if the randomId is valid
  if payload["randomId"] not in STORAGE["users"]["user_data"]:
    # invalid randomId
    data = {}
    data["status"] = "Error"
    data["error_description"] = "Invalid randomId. Please check the randomId and try again."
    return json.dumps(data)
  
  # get the model and cxt from the storage
  model_name = STORAGE["users"]["user_data"][payload["randomId"]]["model"]
  cxt = STORAGE["users"]["user_data"][payload["randomId"]]["cxt"]

  # get the response from the model
  model_obj = STORAGE["static"]["name2mod"][model_name]
  lioness_obj = STORAGE["users"]["user_data"][payload["randomId"]]["lioness"]

  resp_obj, store_obj = utils.get_model_resp(cxt, payload["human_utt"], model_obj, lioness_obj)

  #update internal storage using store_obj
  STORAGE["users"]["user_data"][payload["randomId"]]["lioness"] = store_obj

  # output
  data = {}
  data["status"] = "Success"
  data["randomId"] = payload["randomId"]
  data["response"] = resp_obj
  return json.dumps(data)


@app.route('/reset/', methods=['POST'])
def reset():
  """
  Reset the internal states, as if no user has been connected to the server yet, after the server has been started. You can still keep the models loaded.
  This is useful for debugging.
  """
  global STORAGE

  if SERVER_STATUS != "Ready":
    # server is not yet ready with the initial setup.
    data = {}
    data["status"] = "Error"
    data["error_description"] = "Server is not ready yet. Please wait for a few seconds and try again."
    return json.dumps(data)

  # re-initialize the storage for users
  
  STORAGE["users"]["mod_cxt_used"] = set()
  STORAGE["users"]["user_data"] = {}
  
  # output
  data = {}
  data["status"] = "Success"
  return json.dumps(data)


@app.route('/report_stats/', methods=['POST'])
def report_stats():
  """
  Report basic stats about the current state of the server.
    - # of models, contexts, etc.
    - how many users have connected so far.
    - how many users are remaining, so on.
  This is useful for debugging.
  """

  if SERVER_STATUS != "Ready":
    # initial setup has not been done yet.
    data = {}
    data["status"] = "Error"
    data["error_description"] = "Server is not ready yet. Please wait for a few seconds and try again."
    return json.dumps(data)
  
  # compute all useful stats from STORAGE
  data = {}
  data["status"] = "Success"
  data["server_status"] = SERVER_STATUS
  data["num_models"] = len(STORAGE["static"]["name2mod"])
  data["model_names"] = list(STORAGE["static"]["name2mod"].keys())
  data["num_mod_cxt_used"] = f'{len(STORAGE["users"]["mod_cxt_used"])} / {len(STORAGE["static"]["mod_cxt_all"])}'
  data["num_users_served"] = len(STORAGE["users"]["user_data"])

  # print a sample user data
  if len(STORAGE["users"]["user_data"]) > 0:
    print("Sample user data:")
    print(list(STORAGE["users"]["user_data"].values())[0])
    
  return json.dumps(data)


# run initial setup
initial_setup()


def create_app():
  return app