import json
import os
import re

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from shared.logger import get_logger
from shared.schemas import ExtracaoOutput, MatrizTradeOffs

ARQUIVO_REFERENCIAS = 'output/referencias_pesquisa.txt'


class SearchInput(BaseModel):
    query: str = Field(description="Termo de busca a ser pesquisado na web")


class SearchTool(BaseTool):
    name: str = "Search SerpAPI"
    description: str = "Busca na web e retorna apenas os resultados orgânicos filtrados."
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        api_key = os.getenv("SERPAPI_API_KEY")
        params = {"q": query, "api_key": api_key}

        try:
            response = requests.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            data = response.json()

            filtered_results = []
            if "organic_results" in data:
                for result in data["organic_results"][:10]:
                    filtered_results.append({
                        "title": result.get("title"),
                        "link": result.get("link"),
                        "snippet": result.get("snippet")
                    })

            get_logger().info(f"SEARCH  query='{query}' → {len(filtered_results)} resultados")

            with open(ARQUIVO_REFERENCIAS, "a", encoding="utf-8") as f:
                f.write(f"\n[Termo: '{query}']\n")
                for res in filtered_results:
                    f.write(f"- {res['link']}\n")

            return json.dumps(filtered_results, ensure_ascii=False)

        except requests.exceptions.RequestException as e:
            get_logger().error(f"SEARCH_ERROR  query='{query}' → {e}")
            return json.dumps({"error": str(e)})


search_tool = SearchTool()


class FinalAnswerInput(BaseModel):
    answer: str = Field(description="A resposta final da tarefa")


class FinalAnswerTool(BaseTool):
    name: str = "final_answer"
    description: str = "Retorna a resposta final quando você completou a tarefa."
    args_schema: type[BaseModel] = FinalAnswerInput
    result_as_answer: bool = True

    def _run(self, answer: str) -> str:
        return answer


final_answer_tool = FinalAnswerTool()


# ============================================
# FUNÇÕES AUXILIARES PARA EXTRAIR JSON
# ============================================


# ============================================
# GUARDRAILS - FUNÇÕES NORMAIS
# ============================================

def extrair_json_do_texto(texto: str) -> str:
    """
    Extrai JSON de um texto que pode conter markdown, preâmbulos, etc.
    Tenta múltiplas estratégias de extração.
    """
    if '```json' in texto:
        texto = texto.split('```json')[1].split('```')[0]
    elif '```' in texto:
        texto = texto.split('```')[1].split('```')[0]

    match = re.search(r'\{.*\}', texto, re.DOTALL)
    if match:
        texto = match.group(0)

    return texto.strip()


def limpar_json_para_parse(texto_json: str) -> str:
    """
    Limpa JSON removendo problemas comuns:
    - Backticks markdown dentro de strings
    - Aspas simples em vez de duplas
    """
    texto_json = texto_json.replace('`', '')

    return texto_json


def validar_extracao(resultado):
    """Valida extração de dados com extração robusta de JSON."""
    try:
        texto = resultado.raw if hasattr(resultado, 'raw') else str(resultado)

        texto_json = extrair_json_do_texto(texto)

        texto_json = limpar_json_para_parse(texto_json)

        with open('output/debug_json_extracao.txt', 'w', encoding='utf-8') as f:
            f.write("=== JSON EXTRAÍDO E LIMPO ===\n")
            f.write(texto_json)
            f.write("\n=== FIM ===\n")

        dados_json = json.loads(texto_json)

        fatos_array = None
        campo_usado = None

        if isinstance(dados_json, dict):
            if 'fatos' in dados_json:
                fatos_array = dados_json['fatos']
                campo_usado = 'fatos'
            elif 'ferramentas' in dados_json:
                fatos_array = dados_json['ferramentas']
                campo_usado = 'ferramentas'
            elif 'data' in dados_json:
                fatos_array = dados_json['data']
                campo_usado = 'data'
            elif 'items' in dados_json:
                fatos_array = dados_json['items']
                campo_usado = 'items'
            else:
                campos_existentes = list(dados_json.keys())
                return (False,
                        f"Campo 'fatos' não encontrado. Campos existentes: {campos_existentes}. Use exatamente: {{\"fatos\": [...]}}")
        elif isinstance(dados_json, list):
            fatos_array = dados_json
            campo_usado = 'lista_direta'
        else:
            return (False,
                    f"Formato JSON inválido: deve ser um objeto com campo 'fatos' ou uma lista. Recebido: {type(dados_json)}")

        if campo_usado != 'fatos' and campo_usado != 'lista_direta':
            print(f"[INFO] Normalizando campo '{campo_usado}' para 'fatos'")
            dados_json = {'fatos': fatos_array}
        elif campo_usado == 'lista_direta':
            dados_json = {'fatos': fatos_array}

        extracao = ExtracaoOutput(**dados_json)

        if not extracao.fatos or len(extracao.fatos) == 0:
            return (False, "Nenhum fato extraído. Refaça a pesquisa.")

        return (True, extracao)

    except json.JSONDecodeError as e:
        with open('debug_json_erro.txt', 'w', encoding='utf-8') as f:
            f.write(f"=== ERRO DE PARSING ===\n")
            f.write(f"Erro: {str(e)}\n")
            f.write(f"Posição: linha {e.lineno}, coluna {e.colno}\n")
            f.write(f"\n=== JSON COMPLETO ===\n")
            f.write(texto_json)
            f.write("\n\n=== CARACTERES PRÓXIMOS AO ERRO ===\n")
            if e.colno > 50:
                f.write(f"...{texto_json[e.colno - 50:e.colno + 50]}...\n")
            f.write("\n=== FIM ===\n")

        return (False,
                f"JSON INVÁLIDO. REGRAS CRÍTICAS:\n1. Use APENAS aspas duplas (\"), NUNCA use aspas simples ou backticks (`)\n2. Não use backticks markdown (`) em nenhum lugar do JSON\n3. Todas as strings devem usar aspas duplas\n4. Exemplo correto: \"limitacoes\": [\"A ferramenta tem limite de configuracao\"]\n5. ERRADO: \"limitacoes\": [\"A ferramenta tem `limit` configuracao\"] (sem backticks!)")

    except Exception as e:
        return (False, f"Erro de validação: {str(e)}")


def validar_auditoria(resultado):
    """Valida auditoria SRE com extração robusta de JSON."""
    try:
        texto = resultado.raw if hasattr(resultado, 'raw') else str(resultado)
        texto_json = extrair_json_do_texto(texto)
        texto_json = limpar_json_para_parse(texto_json)

        with open('output/debug_json_auditoria.txt', 'w', encoding='utf-8') as f:
            f.write("=== JSON EXTRAÍDO E LIMPO ===\n")
            f.write(texto_json)
            f.write("\n=== FIM ===\n")

        dados_json = json.loads(texto_json)
        auditoria = MatrizTradeOffs(**dados_json)

        if not auditoria.vencedor_operacional:
            return (False, "Falta definir vencedor operacional")
        if not auditoria.riscos_sre or len(auditoria.riscos_sre) == 0:
            return (False, "Falta identificar riscos SRE")
        if not auditoria.custos_ocultos_finops or len(auditoria.custos_ocultos_finops) == 0:
            return (False, "Falta identificar custos ocultos")
        if not auditoria.recomendacao_final:
            return (False, "Falta a recomendação final")

        return (True, auditoria)

    except json.JSONDecodeError as e:
        return (False,
                f"JSON INVÁLIDO. REGRAS CRÍTICAS:\n1. Use APENAS aspas duplas (\"), NUNCA use aspas simples ou backticks (`)\n2. Não use backticks markdown (`) em nenhum lugar do JSON\n3. Todas as strings devem usar aspas duplas")

    except Exception as e:
        return (False, f"Erro na auditoria: {str(e)}")