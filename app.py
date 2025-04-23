from env import BoardEnv
from agent import Agent
import logging

logging.getLogger().setLevel(logging.INFO)

def main():
    logging.info("Starting app...")

    env = BoardEnv()
    agent = Agent()
    state = env.reset()
    env.render()

    done = False
    while not done:
        action = agent.take_action(state, env.action_space.n)
        new_state, _, done, _ = env.step(action)
        state = new_state
        env.render()

    logging.info("App finished.")

if __name__ == "__main__":
    main()