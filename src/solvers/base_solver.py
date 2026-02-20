class BaseSolver:
    """Classe abstraite pour tous les algorithmes de résolution"""
    def __init__(self, **kwargs):
        pass

    def solve(self, cylinders):
        """
        Doit retourner un tuple: (meilleur_chemin_liste, meilleur_score)
        """
        raise NotImplementedError("La méthode solve() doit être implémentée")