from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from src.ingestao import carregar_teste, carregar_treino_acumulado, carregar_validacao
from src.modelagem import preparar_folds
from src.particionamento import grupos_traco, validar_separacao
from src.utils import carregar_config


class ParticionamentoTest(TestCase):
    def test_reune_idades_e_superplastificante_ausente(self) -> None:
        traço = {
            "cimento": 350, "escoria": 100, "cinza_volante": 0, "agua": 180,
            "superplastificante": 6, "agregado_graudo": 970, "agregado_miudo": 780,
            "idade": 28,
        }
        outros = traço | {"idade": 90, "superplastificante": None}
        inédito = traço | {"cimento": 300}
        dados = pd.DataFrame([traço, outros, inédito])
        grupos = grupos_traco(dados)
        self.assertEqual(grupos.iloc[0], grupos.iloc[1])
        self.assertNotEqual(grupos.iloc[0], grupos.iloc[2])
        with self.assertRaisesRegex(ValueError, "traços em comum"):
            validar_separacao(treino=dados.iloc[:1], teste=dados.iloc[1:])

    def test_artefatos_e_folds_sem_tracos_repetidos(self) -> None:
        cfg = carregar_config()
        treino = carregar_treino_acumulado(cfg)
        if treino is None:
            self.fail("Execute o pipeline para gerar o treino acumulado.")
        validar_separacao(treino=treino, validacao=carregar_validacao(cfg), teste=carregar_teste(cfg))
        features = ["idade"]
        with patch("src.modelagem.votar", return_value=(None, features, {}, "")) as votar:
            folds = preparar_folds(treino, cfg)
        self.assertEqual(len(folds), cfg["avaliacao"]["cv_folds"])
        self.assertEqual(votar.call_count, len(folds))
        for fold, chamada in zip(folds, votar.call_args_list, strict=True):
            validar_separacao(treino=fold.treino, validacao=fold.validacao)
            pd.testing.assert_frame_equal(chamada.args[0], fold.treino)
