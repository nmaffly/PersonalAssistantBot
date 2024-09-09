import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Annotated
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from datetime import datetime
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
import os.path
import datetime as dt
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()

os.environ["ANTHROPIC_API_KEY"] = os.getenv('ANTHROPIC_API_KEY')
os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')

TASK_SCOPES = ["https://www.googleapis.com/auth/tasks"]
CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar"]

def get_tasks_service():
    creds = None
    if os.path.exists('token_tasks.json'):
        creds = Credentials.from_authorized_user_file('token_tasks.json', TASK_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', TASK_SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token_tasks.json', 'w') as token:
            token.write(creds.to_json())
    service = build('tasks', 'v1', credentials=creds)
    return service

def get_calendar_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', CALENDAR_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', CALENDAR_SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    service = build('calendar', 'v3', credentials=creds)
    return service

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

@tool
def schedule_event(summary: str, start_time: str, end_time: str, description: str = '', location: str = '') -> str:
    """Schedule an event in Google Calendar."""

    service = get_calendar_service()
    event = {
        'summary': summary,
        'location': location,
        'description': description,
        'start': {
            'dateTime': start_time,
            'timeZone': 'America/Los_Angeles',
        },
        'end': {
            'dateTime': end_time,
            'timeZone': 'America/Los_Angeles',
        },
    }
    event = service.events().insert(calendarId='primary', body=event).execute()
    return f"Event created: {event.get('htmlLink')}"

@tool
def schedule_recurring_event(
    summary: str,
    start_time: str,
    end_time: str,
    recurrence_rule: str,
    description: str = '',
    location: str = '',
    calendar_id: str = 'primary'
) -> str:
    """Schedule a recurring event in Google Calendar."""

    service = get_calendar_service()
    event = {
        'summary': summary,
        'location': location,
        'description': description,
        'start': {
            'dateTime': start_time,
            'timeZone': 'America/Los_Angeles',
        },
        'end': {
            'dateTime': end_time,
            'timeZone': 'America/Los_Angeles',
        },
        'recurrence': [
            recurrence_rule
        ]
    }
    event = service.events().insert(calendarId=calendar_id, body=event).execute()
    return f"Recurring event created: {event.get('htmlLink')}"

@tool
def list_upcoming_events(num_events: int = 3) -> list:
    """List upcoming events in Google Calendar."""

    service = get_calendar_service()
    now = dt.datetime.now().isoformat() + 'Z'
    events_result = service.events().list(calendarId='primary', timeMin=now,
                                          maxResults=num_events, singleEvents=True,
                                          orderBy='startTime').execute()
    events = events_result.get('items', [])
    return events

@tool
def create_task(title: str, notes: str = '', due: str = None, tasklist: str = '@default') -> str:
    """Create a new task in Google Tasks."""
    service = get_tasks_service()
    task = {
        'title': title,
        'notes': notes,
    }
    if due:
        task['due'] = due
    task = service.tasks().insert(tasklist=tasklist, body=task).execute()
    return f"Task created: {task.get('title')}"

@tool
def list_tasks(tasklist: str = '@default') -> list:
    """List all tasks in a task list."""

    service = get_tasks_service()
    results = service.tasks().list(tasklist=tasklist).execute()
    tasks = results.get('items', [])
    return [{'id': task['id'], 'title': task['title']} for task in tasks]

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_info: str

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result} 

def main():
    llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=1)

    assistant_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are a helpful personal assistant that is always ready to assist the user with the following things:
                      1. Providing a summary of recent news and/or weather using your search capabilities. Provide your answers in a clear and concise manner.
                      2. Scheduling events in the user's Google Calendar, using information provided by the user. Please provide the link to the event as well.
                      3. Listing upcoming events in the user's Google Calendar. Make sure to list the events each time.
                      4. Creating tasks in the user's Google Tasks. Make sure to capture the due date.
                      5. Listing tasks in the user's Google Tasks. Make sure to list the tasks each time.
    
                     If the user asks to schedule an event or task without providing enough necessary information, ask for any missing information, then schedule the event. 
                     If the user asks to do something within Google Calendar or Google Tasks that is not listed above, tell them they must do it manually. """
                "\nCurrent time: {time}.",
            ),
            ("placeholder", "{messages}"),
        ]
    ).partial(time=datetime.now())

    tools = [
        TavilySearchResults(max_results=1),
        schedule_event,
        schedule_recurring_event,
        list_upcoming_events,
        create_task,
        list_tasks
    ]

    assistant_runnable = assistant_prompt | llm.bind_tools(tools)

    builder = StateGraph(State)

    builder.add_node("assistant", Assistant(assistant_runnable))
    builder.add_node("tools", create_tool_node_with_fallback(tools))
    builder.set_entry_point("assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    memory = SqliteSaver.from_conn_string(":memory:")

    graph = builder.compile(checkpointer=memory)

    config = {
        "configurable": {
            # Checkpoints are accessed by thread_id
            "thread_id": "1",
        }
    }
    
    _printed = set()
    while True:
        user_input = input("User: ")
        print("====================================")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        events = graph.stream({"messages": [("user", user_input)]}, config, stream_mode="values")
        # for event in events:
        #     _print_event(event, _printed)
        events_list = list(events)
        last_event = events_list[-1]
        for value in last_event.values():
            print(f"Assistant: {value[-1].content}")
        print("====================================")
            

if __name__ == "__main__":
    main()

