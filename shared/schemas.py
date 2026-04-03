from typing import List
from pydantic import BaseModel, Field


class FatoTecnico(BaseModel):
    ferramenta: str
    arquitetura: str
    limitacoes_conhecidas: List[str]
    fontes: List[str]

class ExtracaoOutput(BaseModel):
    fatos: List[FatoTecnico] = Field(
        description="Lista de fatos extraídos de todas as ferramentas"
    )


class MatrizTradeOffs(BaseModel):
    vencedor_operacional: str = Field(
        description="Qual ferramenta vence em operação"
    )
    riscos_sre: List[str] = Field(
        description="Riscos identificados para SRE"
    )
    custos_ocultos_finops: List[str] = Field(
        description="Custos ocultos identificados"
    )
    recomendacao_final: str = Field(
        description="Recomendação final baseada na análise"
    )