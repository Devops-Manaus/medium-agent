import os
from datetime import datetime
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

llm_qa = LLM(model='ollama/qwen2.5:7b', base_url='http://localhost:11434', temperature=0.0)
llm_writer = LLM(
    model='ollama/deepseek-r1:8b',
    base_url='http://localhost:11434',
    temperature=0.2,
    timeout=1800
)

@CrewBase
class EscritaValidacaoCrew():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config['writer'],
            llm=llm_writer,
            allow_delegation=False,
            verbose=True
        )

    @agent
    def fact_checker(self) -> Agent:
        return Agent(
            config=self.agents_config['fact_checker'],
            llm=llm_qa,
            allow_delegation=False,
            verbose=True
        )

    @task
    def draft_task(self) -> Task:
        return Task(
            config=self.tasks_config['draft_task'],
            agent=self.writer()
        )

    @task
    def formatting_task(self) -> Task:
        return Task(
            config=self.tasks_config['formatting_task'],
            agent=self.writer(),
            context=[self.draft_task()]
        )

    @task
    def validation_task(self) -> Task:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('result', exist_ok=True)
        return Task(
            config=self.tasks_config['validation_task'],
            agent=self.fact_checker(),
            context=[self.formatting_task()],
            output_file=f'result/guia_implementacao_pratico_{ts}.md'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )