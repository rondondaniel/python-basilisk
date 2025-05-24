"""
Simple test to verify the agent and agent_brain classes are working
"""
import logging
from agent import Agent
from env import BoardEnv
from models import EmotionTypes, ActionTypes, CellTypes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_agent_functionality():
    """Test basic agent functionality with the new file structure"""
    try:
        # Create environment and agent separately
        logger.info("Creating environment...")
        env = BoardEnv()
        
        # Create agent without environment
        logger.info("Creating agent...")
        agent = Agent()
        
        # Initialize agent state from environment
        logger.info("Initializing agent state from environment...")
        agent.initialize_state_from_env(env)
        
        # Test observation with explicit environment passing
        logger.info("Testing agent observation with explicit environment passing...")
        observation = agent.observe(env)
        logger.info(f"Observation: {observation}")
        
        # Test emotion setting (doesn't need environment)
        logger.info("Testing emotion setting...")
        emotion_result = agent.set_emotion("HUNGRY")
        logger.info(f"Emotion result: {emotion_result}")
        
        # Test action with explicit environment passing
        logger.info("Testing agent movement with explicit environment passing...")
        action_result = agent.take_action("right", env)
        logger.info(f"Action result: {action_result}")
        
        logger.info("✅ All tests completed successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ Error in test: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting agent functionality test...")
    success = test_agent_functionality()
    if success:
        logger.info("✅ Test completed successfully. The agent code structure is working properly.")
    else:
        logger.error("❌ Test failed. Please check the error messages above.")
