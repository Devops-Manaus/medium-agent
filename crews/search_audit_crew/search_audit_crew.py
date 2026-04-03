from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import ScrapeWebsiteTool

from tools.custom_tools import search_tool, final_answer_tool

llm_researcher = LLM(model='ollama/qwen2.5:7b', base_url='http://localhost:11434', temperature=0.0)
llm_lite = LLM(model='ollama/qwen3.5:latest', base_url='http://localhost:11434', temperature=0.0)


@CrewBase
class PesquisaAuditoriaCrew():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[search_tool, ScrapeWebsiteTool(), final_answer_tool],
            llm=llm_researcher,
            max_iter=15,
            max_retry_limit=2,
            verbose=True
        )

    @agent
    def data_structurer(self) -> Agent:
        return Agent(
            config=self.agents_config['data_structurer'],
            llm=llm_lite,
            tools=[],
            allow_delegation=False,
            verbose=True
        )

    @agent
    def sre_auditor(self) -> Agent:
        return Agent(
            config=self.agents_config['sre_auditor'],
            llm=llm_lite,
            tools=[],
            allow_delegation=False,
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config['research_task'], agent=self.researcher())

    @task
    def deep_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['deep_extraction_task'],
            agent=self.researcher(),
            context=[self.research_task()]
        )

    @task
    def structure_task(self) -> Task:
        return Task(
            config=self.tasks_config['structure_task'],
            agent=self.data_structurer(),
            context=[self.deep_extraction_task()],
        )

    @task
    def audit_task(self) -> Task:
        return Task(
            config=self.tasks_config['audit_task'],
            agent=self.sre_auditor(),
            context=[self.structure_task()],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, process=Process.sequential, verbose=True)