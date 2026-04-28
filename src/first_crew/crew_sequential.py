from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from first_crew.tools.rag_tools import (
    user_rag_tool, 
    item_rag_tool, 
    review_rag_tool, 
    schema_knowledge,
    get_default_llm
)

@CrewBase
class SequentialCrew():
    """Yelp Recommendation Sequential Crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks_sequential.yaml'

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def user_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['user_analyst'], # type: ignore[index]
            tools=[user_rag_tool, review_rag_tool],
            llm=get_default_llm(),
            verbose=True
        )

    @agent
    def item_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['item_analyst'], # type: ignore[index]
            tools=[item_rag_tool, review_rag_tool],
            llm=get_default_llm(),
            verbose=True
        )

    @agent
    def prediction_modeler(self) -> Agent:
        return Agent(
            config=self.agents_config['prediction_modeler'], # type: ignore[index]
            llm=get_default_llm(),
            verbose=True
        )

    @task
    def analyze_user_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_user_task'], # type: ignore[index]
        )

    @task
    def analyze_item_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_item_task'], # type: ignore[index]
        )

    @task
    def predict_review_task(self) -> Task:
        return Task(
            config=self.tasks_config['predict_review_task'], # type: ignore[index]
            output_file='report_sequential.json'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            knowledge_sources=[schema_knowledge],
            embedder={
                "provider": "huggingface",
                "config": {
                    "model": "BAAI/bge-small-en-v1.5"
                }
            },
            verbose=True
        )
