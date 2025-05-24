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
    agent.initialize_state_from_env(env)
    env.render()
    
    # Get initial observation
    observation = agent.observe(env)
    logging.info(f"Initial observation: {observation}")
    
    # Main loop
    done = False
    steps = 0
    max_steps = 20  # Limit the number of steps to avoid infinite loops
    
    while not done and steps < max_steps:
        # Randomly choose a direction for this example
        directions = ["left", "right", "up", "down", "stay"]
        direction = random.choice(directions)
        
        # Take action with explicit environment passing
        logging.info(f"Taking action: {direction}")
        action_result = agent.take_action(direction, env)
        logging.info(f"Result: {action_result}")
        
        # Check if done
        done = "Done: True" in action_result
        steps += 1
        
        # Render environment
        env.render()
        
        # Get observation after action
        observation = agent.observe(env)
        logging.info(f"Observation: {observation}")
        
        # Optionally process a query with environment context
        if steps % 5 == 0:  # Every 5 steps
            response = agent.process_query("What should I do next?", env)
            logging.info(f"Agent's suggestion: {response['output']}")
    
    logging.info(f"App finished after {steps} steps.")

if __name__ == "__main__":
    main()