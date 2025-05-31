from env import BoardEnv
from agent import Agent
from models import ActionTypes
import logging
import time

logging.getLogger().setLevel(logging.INFO)

def main():
    logging.info("Starting app...")

    # Create environment and agent separately
    env = BoardEnv()
    agent = Agent()
    
    # Initialize agent state from environment
    current_observation = env.reset()
    logging.info("Initial State:\n%s" % current_observation)
    logging.info("---------------------------------------------------------------")

    # Main loop
    done = False
    while not done:
        
        action = agent.take_action(current_observation)
        new_observation, reward, done, info = env.step(action)
        logging.info("Action: %s; Reward: %.4f" % (action, reward))
        logging.info("Next State:\n%s" % new_observation)
        logging.info("---------------------------------------------------------------")
        time.sleep(0.5)
        
        # Get observation after action
        current_observation = new_observation

    logging.info(f"Episode finished.")

if __name__ == "__main__":
    main()