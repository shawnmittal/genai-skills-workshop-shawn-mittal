import os
import logging
import vertexai
from typing import Optional
from vertexai.generative_models import GenerativeModel
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse


logger = logging.getLogger(__name__)

GCP_PROJECT = os.getenv("GCP_PROJECT")
GCP_REGION = os.getenv("GCP_REGION")


def is_address_in_us(project_id: str, location: str, user_query: str) -> bool:
    """Checks if the addresses in a user query are in the United States.

    Args:
        project_id: The Google Cloud project ID.
        location: The Google Cloud location (e.g., "us-central1").
        user_query: The user's query string containing addresses.

    Returns:
        True if model determines all addresses are in the US, False otherwise.
    """
    try:
        vertexai.init(project=project_id, location=location)
        model = GenerativeModel("gemini-2.5-flash")

        prompt = (
            'Are the following addresses in the user query all located in the '
            'United States of America? Please answer with only the word "yes" '
            f'or "no". User Query: "{user_query}"'
        )
        response = model.generate_content(prompt)
        text_response = response.text.strip().lower()

        return text_response == 'yes'

    except Exception as e:
        print(f"An error occurred while checking address location: {e}")

    return False


def is_user_query_mean(project_id: str,
                       location: str, user_query: str) -> bool:
    """Determines if a user query could be considered malicious or mean.

    Args:
        project_id: The Google Cloud project ID.
        location: The Google Cloud location (e.g., "us-central1").
        user_query: The user's query string.

    Returns:
        True if the model determines the query is mean, False otherwise.
    """
    try:
        vertexai.init(project=project_id, location=location)
        model = GenerativeModel("gemini-2.5-flash")

        prompt = (
            'Could the user query be construed as malicious or mean? '
            'Please answer with only the word "yes" or "no". User Query: '
            f'"{user_query}"'
        )
        response = model.generate_content(prompt)
        text_response = response.text.strip().lower()

        return text_response == 'yes'

    except Exception as e:
        print(f"An error occurred during query safety check: {e}")

    return False


def user_prompt_log_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Logs the content of the user's latest prompt.

    Args:
        callback_context: The context of the agent executing the callback.
        llm_request: The request object sent to the LLM.

    Returns:
        LlmResponse or None.
    """
    if llm_request.contents:
        last = llm_request.contents[-1]
        if last.role == "user" and last.parts and last.parts[0].text:
            user_text = last.parts[0].text.strip()
            logger.info(f"[{callback_context.agent_name}] USER >> {user_text}")

    return None


def model_response_log_callback(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Logs the content of the model's response.

    Args:
        callback_context: The context of the agent executing the callback.
        llm_response: The response object received from the LLM.

    Returns:
        LlmResponse or None.
    """
    if llm_response.content and llm_response.content.parts:
        txt = llm_response.content.parts[0].text
        if txt:
            logger.info(
                f"[{callback_context.agent_name}] MODEL >> {txt.strip()}"
            )


def user_query_check_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """Performs moderation checks on the user's query.

    Checks for non-US addresses and harmful content. If a check fails,
    it returns a pre-canned LlmResponse to stop further processing.

    Args:
        callback_context: The context of the agent executing the callback.
        llm_request: The request object containing the user's query.

    Returns:
        An LlmResponse to short-circuit the chain if moderation fails,
        otherwise None.
    """
    try:
        last = llm_request.contents[-1]

        if last.role == "user" and last.parts and last.parts[0].text:
            user_text = last.parts[0].text.strip()
            if not is_address_in_us(
                project_id=GCP_PROJECT,
                location=GCP_REGION,
                user_query=user_text,
            ):
                return LlmResponse(
                    content={
                        "role": "model",
                        "parts": [
                            {
                                "text": "Message contains non-US addresses, "
                                "please only query for US addresses."
                            }
                        ],
                    }
                )

            if is_user_query_mean(
                project_id=GCP_PROJECT,
                location=GCP_REGION,
                user_query=user_text,
            ):
                return LlmResponse(
                    content={"role": "model", "parts": [{"text": "Be nice."}]}
                )

    except Exception as e:
        logger.error(f"Woops:\n{e}")

    return None


def chained_before_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """A chained 'before' callback that runs multiple checks in sequence.

    It first runs a moderation check. If the moderation check returns a
    response, this function immediately returns it. Otherwise, it proceeds
    to log the user's input.

    Args:
        callback_context: The context of the agent executing the callback.
        llm_request: The request object to be processed.

    Returns:
        An LlmResponse if moderation fails, otherwise None.
    """

    # 1. Moderation check
    moderation_result = user_query_check_callback(
        callback_context, llm_request
    )
    if moderation_result is not None:
        return moderation_result

    # 2. Log user input
    user_prompt_log_callback(callback_context, llm_request)

    return None
