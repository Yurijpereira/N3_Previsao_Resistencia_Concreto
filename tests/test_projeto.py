from unittest import TestCase
from unittest.mock import patch

import numpy as np
from sklearn.dummy import DummyRegressor

from src.dados import ALVO, ENTRADAS, carregar_dados, id_mistura
from src.modelos import separar_treino_teste
from src.prever import prever


class TesteProjeto(TestCase):
    def test_treino_e_teste_nao_compartilham_tracos(self) -> None:
        dados = carregar_dados()
        grupos = id_mistura(dados)
        treino, teste = separar_treino_teste(dados[ENTRADAS], dados[ALVO], grupos)
        self.assertEqual(len(treino) + len(teste), len(dados))
        self.assertFalse(set(grupos[treino]) & set(grupos[teste]))

    def test_entradas_invalidas_sao_rejeitadas_antes_do_modelo(self) -> None:
        casos = (
            ({"cimento": 0}, "cimento"),
            ({"agua": -1}, "agua"),
            ({"idade": 0}, "idade"),
            ({"idade": 28.5}, "idade"),
            ({"escoria": float("nan")}, "escoria"),
            ({"agua": float("inf")}, "agua"),
            ({"atributo_desconhecido": 10}, "desconhecido"),
        )
        with patch("src.prever.carregar_modelo") as carregar:
            for entradas, motivo in casos:
                with self.subTest(entradas=entradas), self.assertRaisesRegex(ValueError, motivo):
                    prever(**entradas)
            carregar.assert_not_called()

    def test_previsao_informa_faixa_e_extrapolacao(self) -> None:
        modelo = DummyRegressor(strategy="constant", constant=35)
        modelo.fit(np.zeros((1, len(ENTRADAS))), [35])
        with patch("src.prever.carregar_modelo", return_value={
            "modelo": modelo, "entradas": ENTRADAS, "faixas": {"cimento": (102, 540)},
            "margem_80": 5.8,
        }):
            resultado = prever(cimento=600)
        self.assertEqual(resultado["resistencia_MPa"], 35.0)
        self.assertEqual(resultado["intervalo_80"], (29.2, 40.8))
        self.assertIn("fora da faixa", resultado["avisos"][0])
