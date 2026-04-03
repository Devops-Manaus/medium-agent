import glob
import json
import os
import re
import traceback
from datetime import datetime as dt
from crewai.flow.flow import Flow, start, listen
from dotenv import load_dotenv

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

from crews.search_audit_crew.search_audit_crew import PesquisaCrew, AuditoriaCrew
from crews.write_validate_crew.write_validate_crew import EscritaValidacaoCrew
from shared.logger import setup_run_logger, get_logger
from shared.schemas import ExtracaoOutput
from tools.custom_tools import validar_extracao, validar_auditoria

load_dotenv()

ARQUIVO_REFERENCIAS = 'output/referencias_pesquisa.txt'
console = Console()


def _parse_ferramentas(ferramentas_str: str) -> list[str]:
    partes = re.split(r'\s+(?:e|vs\.?|versus|and)\s+|,\s*', ferramentas_str, flags=re.IGNORECASE)
    return [p.strip() for p in partes if p.strip()]


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

        if os.path.exists(ARQUIVO_REFERENCIAS):
            os.remove(ARQUIVO_REFERENCIAS)

        ferramentas = _parse_ferramentas(self.inputs['ferramentas_alvo'])
        log.info(f"[1/3] Pesquisa iniciada | {len(ferramentas)} ferramenta(s): {ferramentas}")
        console.print(f"\n[1/3] Pesquisando [bold yellow]{len(ferramentas)}[/bold yellow] ferramenta(s): {ferramentas}")

        todos_fatos = []

        for i, ferramenta in enumerate(ferramentas, 1):
            console.print(f"\n  [{i}/{len(ferramentas)}] Pesquisando: [bold cyan]{ferramenta}[/bold cyan]...")
            log.info(f"  Iniciando pesquisa de: {ferramenta}")

            inputs_ferramenta = {**self.inputs, 'ferramenta_atual': ferramenta}

            try:
                resultado = PesquisaCrew().crew().kickoff(inputs=inputs_ferramenta)
                self._salvar_metricas(resultado, f"Pesquisa:{ferramenta}")

                task_estrutura = resultado.tasks_output[-1]
                ok, dados = validar_extracao(task_estrutura)

                if ok:
                    todos_fatos.extend(dados.fatos)
                    log.info(f"  {ferramenta}: extração validada com sucesso")
                else:
                    log.warning(f"  {ferramenta}: extração inválida — {dados} | usando raw como fallback")
                    todos_fatos_raw = task_estrutura.raw
                    todos_fatos.append({
                        "ferramenta": ferramenta,
                        "arquitetura": todos_fatos_raw[:500] if todos_fatos_raw else "Não disponível",
                        "limitacoes_conhecidas": ["Dados insuficientes — validação falhou"],
                        "fontes": []
                    })

            except Exception as e:
                log.error(f"  {ferramenta}: pesquisa falhou — {e}")
                log.error(traceback.format_exc())
                console.print(f"  [bold red]Pesquisa de {ferramenta} falhou: {e}[/bold red]")
                todos_fatos.append({
                    "ferramenta": ferramenta,
                    "arquitetura": "Não disponível — erro na pesquisa",
                    "limitacoes_conhecidas": ["Dados insuficientes — erro na execução"],
                    "fontes": []
                })

        duracao_pesquisa = (dt.now() - t0).total_seconds()
        log.info(f"[1/3] Pesquisa concluída em {duracao_pesquisa:.1f}s | {len(todos_fatos)} ferramenta(s) coletadas")

        dados_agregados = ExtracaoOutput(fatos=todos_fatos)
        dados_json = json.dumps(dados_agregados.model_dump(), ensure_ascii=False)
        log.info(f"dados_agregados: {dados_json[:600]}")

        self.state['dados_estruturados'] = dados_json
        return dados_json

    @listen(executar_pesquisa)
    def executar_auditoria(self, dados_estruturados):
        log = get_logger()
        t0 = dt.now()
        console.print("\n[2/3] Iniciando Auditoria SRE...")
        log.info(f"[2/3] Auditoria iniciada | dados={len(dados_estruturados)} chars")

        inputs_auditoria = {
            **self.inputs,
            'dados_estruturados': dados_estruturados,
        }

        auditoria_json = dados_estruturados  # fallback

        try:
            resultado_auditoria = AuditoriaCrew().crew().kickoff(inputs=inputs_auditoria)
            self._salvar_metricas(resultado_auditoria, "Auditoria")

            ok, auditoria = validar_auditoria(resultado_auditoria.tasks_output[-1])
            if ok:
                auditoria_json = json.dumps(auditoria.model_dump(), ensure_ascii=False)
                log.info(f"Auditoria validada | vencedor={auditoria.vencedor_operacional}")
            else:
                log.warning(f"Auditoria inválida — {auditoria} | usando raw")
                auditoria_json = resultado_auditoria.raw

        except Exception as e:
            log.error(f"Auditoria falhou: {e}")
            log.error(traceback.format_exc())
            console.print(f"[bold red]Auditoria falhou: {e}[/bold red]")

        duracao = (dt.now() - t0).total_seconds()
        log.info(f"[2/3] Auditoria concluída em {duracao:.1f}s")

        dados_auditoria = (
            f"=== DADOS TÉCNICOS EXTRAÍDOS ===\n{dados_estruturados}"
            f"\n\n=== AUDITORIA SRE ===\n{auditoria_json}"
        )
        self.state['dados_auditoria'] = dados_auditoria
        return dados_auditoria

    @listen(executar_auditoria)
    def executar_escrita(self, dados_auditoria):
        log = get_logger()
        t0 = dt.now()
        console.print("\n[3/3] Iniciando Escrita e Validação...")
        log.info(f"[3/3] Escrita iniciada | dados_auditoria={len(dados_auditoria)} chars")
        log.info(f"dados_auditoria preview: {dados_auditoria[:600]}")

        inputs_escrita = {**self.inputs, 'dados_auditoria': dados_auditoria}

        crew_escrita = EscritaValidacaoCrew().crew()
        resultado_final = crew_escrita.kickoff(inputs=inputs_escrita)

        self._salvar_metricas(resultado_final, "Escrita")
        self._anexar_referencias()

        self.state['status'] = 'concluido'
        self.state['resultado_final'] = resultado_final.raw

        duracao = (dt.now() - t0).total_seconds()
        log.info(f"[3/3] Escrita concluída em {duracao:.1f}s")
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
