import numpy as np
import filtri


def test_konvolucija_basic():
    slika = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float32)

    jedro = np.ones((3, 3), dtype=np.float32)

    rezultat = filtri.konvolucija(slika, jedro)

    assert rezultat.shape == slika.shape


def test_sobel_output_shape():
    slika = np.random.rand(10, 10).astype(np.float32)
    rezultat = filtri.sobel_vertikalno(slika, np.float32(0.1), (0, 0, 1))

    assert rezultat.shape == (10, 10, 3)


def test_koticki_output():
    slika = np.random.rand(10, 10).astype(np.float32)
    rezultat = filtri.poisci_koticke_rotiranih_kvadratov(slika)

    assert rezultat.shape[2] == 4


def test_znak_a():
    slika = np.random.rand(10, 10).astype(np.float32)
    rezultat = filtri.poisci_znak_a(slika)

    assert rezultat.shape == slika.shape


def test_orientacija():
    slika = np.random.rand(10, 10).astype(np.float32)
    kot = filtri.oceni_orientacijo_horizonta(slika)

    assert isinstance(kot, float)
