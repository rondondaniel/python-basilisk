"""
Test script to verify the Agent with dependency injection architecture
"""
import logging
# Fix path for imports
from tests.context import *
from agent import Agent
from env import BoardEnv
from models import EmotionTypes, ActionTypes, CellTypes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_agent_with_dependency_injection():
    """Test the Agent with proper dependency injection for the environment"""
    try:
        # Create environment and agent separately
        logger.info("Creating environment...")
        env = BoardEnv()
        
        # Reset environment to get initial state
        logger.info("Resetting environment...")
        initial_state = env.reset()
        
        # Create agent without environment
        logger.info("Creating agent (without environment)...")
        agent = Agent()
        
        # Initialize agent state from environment
        logger.info("Initializing agent state from environment...")
        agent.initialize_state_from_env(env)
        
        # Test observation with explicit environment passing
        logger.info("Testing agent observation with explicit environment passing...")
        observation = agent.observe(env)
        logger.info(f"Observation: {observation}")
        
        # Test action with explicit environment passing
        logger.info("Testing agent movement with explicit environment passing...")
        action_result = agent.take_action("right", env)
        logger.info(f"Action result: {action_result}")
        
        # Test emotion setting (doesn't need environment)
        logger.info("Testing emotion setting...")
        emotion_result = agent.set_emotion("HUNGRY")
        logger.info(f"Emotion result: {emotion_result}")
        
        # Test processing a query with environment context
        logger.info("Testing LLM processing with environment context...")
        query = "Where am I and what's around me?"
        llm_result = agent.process_query(query, env)
        logger.info(f"Query result: {llm_result}")
        
        # Test processing a query without environment context
        logger.info("Testing LLM processing without environment context...")
        query = "What is your emotion?"
        llm_result_no_env = agent.process_query(query)
        logger.info(f"Query result (no env): {llm_result_no_env}")
        
        logger.info("✅ All dependency injection tests completed successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ Error in dependency injection test: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting agent dependency injection test...")
    success = test_agent_with_dependency_injection()
    if success:
        logger.info("✅ Dependency injection test completed successfully.")
    else:
        logger.error("❌ Dependency injection test failed. Please check the error messages above.")
