import glob
import json
import os
import traceback
from datetime import datetime as dt
from crewai.flow.flow import Flow, start, listen
from dotenv import load_dotenv

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

from crews.search_audit_crew.search_audit_crew import PesquisaAuditoriaCrew
from crews.write_validate_crew.write_validate_crew import EscritaValidacaoCrew
from shared.logger import setup_run_logger, get_logger

load_dotenv()

ARQUIVO_REFERENCIAS = 'output/referencias_pesquisa.txt'
console = Console()

class TechAnalysisFlow(Flow):
    def __init__(self):
        super().__init__()
        self._run_id = dt.now().strftime("%Y%m%d_%H%M%S")
        self._start_time = dt.now()
        setup_run_logger(self._run_id)
        self._coletar_inputs()

    def _coletar_inputs(self):
        console.print(Panel.fit("Pipeline de Análise Arquitetural com CrewAI", style="bold blue"))

        ferramentas = Prompt.ask("[bold green]Ferramentas alvo[/bold green] (Ex: Snowflake e BigQuery)")
        contexto = Prompt.ask("[bold green]Contexto da análise[/bold green] (Ex: Armazenamento e processamento analítico)")
        requisitos = Prompt.ask("[bold green]Requisitos técnicos[/bold green] (Ex: Modelo de pricing, custos ocultos de rede)")

        self.inputs = {
            'ferramentas_alvo': ferramentas.strip(),
            'contexto': contexto.strip(),
            'requisito_tecnico': requisitos.strip(),
        }

        log = get_logger()
        log.info(
            f"Inputs registrados | ferramentas_alvo={self.inputs['ferramentas_alvo']} "
            f"| contexto={self.inputs['contexto']} "
            f"| requisito_tecnico={self.inputs['requisito_tecnico']}"
        )
        console.print("\n[bold cyan]Inputs registrados. Iniciando fluxo...[/bold cyan]\n")

    @start()
    def executar_pesquisa(self):
        log = get_logger()
        t0 = dt.now()
        console.print(f"\n[1/2] Iniciando Pesquisa e Auditoria para: [bold yellow]{self.inputs['ferramentas_alvo']}[/bold yellow]...")
        log.info(f"[1/2] Etapa Pesquisa iniciada")

        if os.path.exists(ARQUIVO_REFERENCIAS):
            os.remove(ARQUIVO_REFERENCIAS)

        self.state['dados_auditoria'] = "Nenhum dado extraído."

        try:
            crew_pesquisa = PesquisaAuditoriaCrew().crew()
            resultado_pesquisa = crew_pesquisa.kickoff(inputs=self.inputs)

            self._salvar_metricas(resultado_pesquisa, "Pesquisa")

            if hasattr(resultado_pesquisa, 'tasks_output') and len(resultado_pesquisa.tasks_output) >= 2:
                fatos_estruturados = resultado_pesquisa.tasks_output[-2].raw
                auditoria_final = resultado_pesquisa.tasks_output[-1].raw
                self.state['dados_auditoria'] = f"=== DADOS TÉCNICOS EXTRAÍDOS ===\n{fatos_estruturados}\n\n=== AUDITORIA SRE ===\n{auditoria_final}"

                log.info(f"Task outputs recebidos ({len(resultado_pesquisa.tasks_output)} tasks)")
                for i, t in enumerate(resultado_pesquisa.tasks_output):
                    log.info(f"  tasks_output[{i}]: {str(t.raw)[:400]}")
            else:
                self.state['dados_auditoria'] = resultado_pesquisa.raw
                log.info(f"Pesquisa raw output: {str(resultado_pesquisa.raw)[:400]}")

        except Exception as e:
            log.error(f"[AVISO] Etapa Pesquisa falhou: {e}")
            log.error(traceback.format_exc())
            console.print(f"\n[bold red][AVISO] A auditoria falhou ou o Guardrail abortou: {e}[/bold red]")
            console.print("[yellow]A resgatar dados parciais da extração e forçar a escrita do artigo...[/yellow]")

            fatos_recuperados = "Dados estruturados não disponíveis."
            if os.path.exists('output/debug_json_extracao.txt'):
                with open('output/debug_json_extracao.txt', 'r', encoding='utf-8') as f:
                    fatos_recuperados = f.read()
                log.info(f"Dados recuperados de debug_json_extracao.txt ({len(fatos_recuperados)} chars)")

            self.state['dados_auditoria'] = f"=== DADOS TÉCNICOS EXTRAÍDOS (RECUPERADOS) ===\n{fatos_recuperados}\n\n=== AUDITORIA SRE ===\n[PREENCHER MANUALMENTE: A auditoria SRE falhou na formatação e foi ignorada]"

        duracao = (dt.now() - t0).total_seconds()
        log.info(f"[1/2] Etapa Pesquisa concluída em {duracao:.1f}s")
        return self.state['dados_auditoria']

    @listen(executar_pesquisa)
    def executar_escrita(self, dados_auditoria):
        log = get_logger()
        t0 = dt.now()
        console.print("\n[2/2] Iniciando Escrita e Validação...")
        log.info(f"[2/2] Etapa Escrita iniciada | dados_auditoria={len(dados_auditoria)} chars")
        log.info(f"dados_auditoria preview: {dados_auditoria[:600]}")

        inputs_escrita = self.inputs.copy()
        inputs_escrita['dados_auditoria'] = dados_auditoria

        crew_escrita = EscritaValidacaoCrew().crew()
        resultado_final = crew_escrita.kickoff(inputs=inputs_escrita)

        self._salvar_metricas(resultado_final, "Escrita")
        self._anexar_referencias()

        self.state['status'] = 'concluido'
        self.state['resultado_final'] = resultado_final.raw

        duracao = (dt.now() - t0).total_seconds()
        log.info(f"[2/2] Etapa Escrita concluída em {duracao:.1f}s")
        log.info(f"resultado_final preview: {str(resultado_final.raw)[:600]}")
        return self.state['resultado_final']

    @listen(executar_escrita)
    def finalizar_fluxo(self, resultado_final):
        log = get_logger()
        duracao_total = (dt.now() - self._start_time).total_seconds()
        log.info(f"Fluxo encerrado. Duração total: {duracao_total:.1f}s")
        console.print("\n[bold green]Fluxo encerrado. Artigo gerado (com placeholders, se necessário).[/bold green]")
        return resultado_final

    def _salvar_metricas(self, resultado, etapa):
        try:
            usage_metrics = {
                "timestamp": dt.now().isoformat(),
                "topic": self.inputs['ferramentas_alvo'],
                "etapa": etapa,
                "usage": resultado.token_usage.dict() if hasattr(resultado, 'token_usage') else "N/A"
            }
            os.makedirs('output', exist_ok=True)
            with open('output/metrics_usage.json', 'a', encoding='utf-8') as f:
                json.dump(usage_metrics, f, indent=4, ensure_ascii=False)
                f.write("\n")
        except Exception as e:
            console.print(f"[bold red]Erro ao salvar métricas da etapa {etapa}: {e}[/bold red]")

    def _anexar_referencias(self):
        if not os.path.exists(ARQUIVO_REFERENCIAS):
            return
        arquivos = glob.glob('result/guia_implementacao_pratico_*.md')
        if not arquivos:
            return
        arquivo_saida = max(arquivos, key=os.path.getmtime)
        with open(arquivo_saida, "a", encoding="utf-8") as out_file:
            out_file.write("\n\n## Referências Consultadas\n")
            with open(ARQUIVO_REFERENCIAS, "r", encoding="utf-8") as ref_file:
                out_file.write(ref_file.read())