import os
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
class HierarchicalCrew():
    """Hierarchical Crew for Yelp Prediction"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks_hierarchical.yaml'

    @agent
    def user_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['user_analyst'], # type: ignore[index]
            tools=[user_rag_tool, review_rag_tool],
            llm=get_default_llm(),
            verbose=True,
            allow_delegation=False
        )

    @agent
    def item_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['item_analyst'], # type: ignore[index]
            tools=[item_rag_tool, review_rag_tool],
            llm=get_default_llm(),
            verbose=True,
            allow_delegation=False
        )

    @agent
    def prediction_modeler(self) -> Agent:
        return Agent(
            config=self.agents_config['prediction_modeler'], # type: ignore[index]
            llm=get_default_llm(),
            verbose=True,
            allow_delegation=False
        )

    @agent
    def prediction_manager(self) -> Agent:
        return Agent(
            config=self.agents_config['prediction_manager'], # type: ignore[index]
            llm=get_default_llm(),
            verbose=True,
            allow_delegation=True
        )

    @task
    def hierarchical_predict_task(self) -> Task:
        return Task(
            config=self.tasks_config['hierarchical_predict_task'], # type: ignore[index]
            output_file='report_hierarchical.json'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.user_analyst(), self.item_analyst(), self.prediction_modeler()],
            tasks=[self.hierarchical_predict_task()],
            process=Process.hierarchical,
            manager_agent=self.prediction_manager(),
            knowledge_sources=[schema_knowledge],
            embedder={
                "provider": "huggingface",
                "config": {
                    "model": "BAAI/bge-small-en-v1.5"
                }
            },
            verbose=True
        )
