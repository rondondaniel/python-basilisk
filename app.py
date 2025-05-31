from env import BoardEnv
from agent import Agent
from models import ActionTypes
import logging
import random

logging.getLogger().setLevel(logging.INFO)

def main():
    logging.info("Starting app...")

    # Create environment and agent separately
    env = BoardEnv()
    agent = Agent()
    
    # Initialize agent state from environment
    current_observation = env.reset()
    
    # Main loop
    done = False
    while not done:
        
        action = agent.take_action(current_observation)
        new_observation, reward, done, info = env.step(action)
        logger.info("Action: %s; Reward: %.4f" % (action, reward))
        logger.info("Next State: %s" % new_observation)
        logger.info("---------------------------------------------------------------")
        
        # Render environment
        env.render()
        
        # Get observation after action
        current_observation = new_observation
        
        # Optionally process a query with environment context
        response = agent.process_query("What should I do next?", env)
        logging.info(f"Agent's suggestion: {response['output']}")
    
    logging.info(f"Episode finished.")

if __name__ == "__main__":
    main()