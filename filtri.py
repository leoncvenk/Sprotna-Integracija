import cv2 as cv
import numpy as np
from pathlib import Path
BASE = Path(__file__).parent
UTILS = BASE / ".utils"

def konvolucija(slika: np.ndarray, jedro: np.ndarray) -> np.ndarray:
    # normalizacija vhoda np.float32
    slika = slika.astype(np.float32)
    jedro = jedro.astype(np.float32)

    # jedro mora biti lihih dimenzij, ndim pove koliko dimenzij ima slika
    if jedro.ndim != 2:
        raise ValueError("Jedro mora biti 2D matrika.")
    if jedro.shape[0] % 2 == 0 or jedro.shape[1] % 2 == 0: #.shape[0] = višina , [1] = širina
        raise ValueError("Jedro mora imeti lihe dimenzije.")

    # velikost jedra in padding
    # kh = višina jedra | kw = širina jedra
    kh, kw = jedro.shape
    # pove, koliko pikslov moraš dodati okoli slike, da lahko filter uporabiš tudi na robovih
    pad_y = kh // 2 # // pomeni celoštevilsko deljenje 3//2=1
    pad_x = kw // 2

    # za pravo konvolucijo jedro obrnemo za 180°
    # zgornji levi element gre v spodnji desni kot itd
    jedro_obrnjeno = np.zeros_like(jedro, dtype=np.float32)
    for y in range(kh):
        for x in range(kw):
            jedro_obrnjeno[y, x] = jedro[kh - 1 - y, kw - 1 - x]

    # =========================
    # SIVINSKA SLIKA
    # =========================
    if slika.ndim == 2:
        # višina, širina
        h, w = slika.shape

        # ročni replicate padding, ustvarjanje razširjene slike
        razsirjena = np.zeros((h + 2 * pad_y, w + 2 * pad_x), dtype=np.float32)

        # sredina
        for y in range(h):
            for x in range(w):
                razsirjena[y + pad_y, x + pad_x] = slika[y, x]

        # zgornji in spodnji rob
        for y in range(pad_y):
            for x in range(w):
                razsirjena[y, x + pad_x] = slika[0, x]
                razsirjena[h + pad_y + y, x + pad_x] = slika[h - 1, x]

        # levi in desni rob
        for y in range(h):
            for x in range(pad_x):
                razsirjena[y + pad_y, x] = slika[y, 0]
                razsirjena[y + pad_y, w + pad_x + x] = slika[y, w - 1]

        # vogali
        for y in range(pad_y):
            for x in range(pad_x):
                razsirjena[y, x] = slika[0, 0]
                razsirjena[y, w + pad_x + x] = slika[0, w - 1]
                razsirjena[h + pad_y + y, x] = slika[h - 1, 0]
                razsirjena[h + pad_y + y, w + pad_x + x] = slika[h - 1, w - 1]

        izhod = np.zeros((h, w), dtype=np.float32)

        # izračun vogalov
        # za vsak izhodni piksel (y, x):
        # 1. vzameš ustrezno okno iz razširjene slike
        # 2. ga pomnožiš element po element z jedrom
        # 3. vse rezultate sešteješ
        # 4. vsoto shraneš v izhod
        for y in range(h):
            for x in range(w):
                vsota = 0.0
                for ky in range(kh):
                    for kx in range(kw):
                        vsota += razsirjena[y + ky, x + kx] * jedro_obrnjeno[ky, kx]
                izhod[y, x] = vsota

        return izhod

    # =========================
    # BARVNA SLIKA
    # =========================
    elif slika.ndim == 3:
        h, w, c = slika.shape
        izhod = np.zeros((h, w, c), dtype=np.float32)

        for kanal in range(c):
            kanal_slika = slika[:, :, kanal]

            # ročni replicate padding za posamezen kanal
            razsirjena = np.zeros((h + 2 * pad_y, w + 2 * pad_x), dtype=np.float32)

            # sredina
            for y in range(h):
                for x in range(w):
                    razsirjena[y + pad_y, x + pad_x] = kanal_slika[y, x]

            # zgornji in spodnji rob
            for y in range(pad_y):
                for x in range(w):
                    razsirjena[y, x + pad_x] = kanal_slika[0, x]
                    razsirjena[h + pad_y + y, x + pad_x] = kanal_slika[h - 1, x]

            # levi in desni rob
            for y in range(h):
                for x in range(pad_x):
                    razsirjena[y + pad_y, x] = kanal_slika[y, 0]
                    razsirjena[y + pad_y, w + pad_x + x] = kanal_slika[y, w - 1]

            # vogali
            for y in range(pad_y):
                for x in range(pad_x):
                    razsirjena[y, x] = kanal_slika[0, 0]
                    razsirjena[y, w + pad_x + x] = kanal_slika[0, w - 1]
                    razsirjena[h + pad_y + y, x] = kanal_slika[h - 1, 0]
                    razsirjena[h + pad_y + y, w + pad_x + x] = kanal_slika[h - 1, w - 1]

            # konvolucija po kanalu
            for y in range(h):
                for x in range(w):
                    vsota = 0.0
                    for ky in range(kh):
                        for kx in range(kw):
                            vsota += razsirjena[y + ky, x + kx] * jedro_obrnjeno[ky, kx]
                    izhod[y, x, kanal] = vsota

        return izhod

    else:
        raise ValueError("Slika mora biti sivinska (H,W) ali barvna (H,W,C).")

def sobel_vertikalno(slika: np.ndarray, max_gradient: np.float32, barva: tuple) -> np.ndarray:
    # tuple je zaporedje v oklepajih (B, G, R)
    # delamo z np.float32
    slika = slika.astype(np.float32)

    # če je slika sivinska, jo pripravimo za Sobel,
    # potem pa jo razširimo v barvno za izhod
    if slika.ndim == 2:
        siva = slika
        izhod = np.stack((slika, slika, slika), axis=2)
    elif slika.ndim == 3:
        # če je vhod barven, za detekcijo robov naredimo sivinsko sliko
        siva = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
        izhod = slika.copy()
    else:
        raise ValueError("Slika mora biti sivinska ali barvna.")

    # Sobel po x -> vertikalni robovi
    gradient_x = cv.Sobel(siva, cv.CV_32F, 1, 0, ksize=3)

    # moč roba = absolutna vrednost odziva
    moc_roba = np.abs(gradient_x)

    # maska vseh pikslov, kjer je rob dovolj močan
    maska = moc_roba > max_gradient

    # barvanje robov
    izhod[maska] = barva

    return izhod

def poisci_koticke_rotiranih_kvadratov(slika: np.ndarray) -> np.ndarray:
    # vedno delamo z np.float32
    slika = slika.astype(np.float32)

    # če je slika barvna, jo pretvorimo v sivinsko
    if slika.ndim == 3:
        siva = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
    elif slika.ndim == 2:
        siva = slika
    else:
        raise ValueError("Slika mora biti sivinska ali barvna.")

    # 4 filtri 3x3 za kotičke rotiranega kvadrata
    # kanali: Z, D, L, S

    # Z = zgornji kotiček
    filter_Z = np.array([
        [ 1,  1,  1],
        [ 0,  1,  0],
        [-1, -1, -1]
    ], dtype=np.float32)

    # D = desni kotiček
    filter_D = np.array([
        [-1,  0,  1],
        [-1,  1,  1],
        [-1,  0,  1]
    ], dtype=np.float32)

    # L = levi kotiček
    filter_L = np.array([
        [ 1,  0, -1],
        [ 1,  1, -1],
        [ 1,  0, -1]
    ], dtype=np.float32)

    # S = spodnji kotiček
    filter_S = np.array([
        [-1, -1, -1],
        [ 0,  1,  0],
        [ 1,  1,  1]
    ], dtype=np.float32)

    odziv_Z = cv.filter2D(siva, cv.CV_32F, filter_Z)
    odziv_D = cv.filter2D(siva, cv.CV_32F, filter_D)
    odziv_L = cv.filter2D(siva, cv.CV_32F, filter_L)
    odziv_S = cv.filter2D(siva, cv.CV_32F, filter_S)

    rezultat = np.stack((odziv_Z, odziv_D, odziv_L, odziv_S), axis=2).astype(np.float32)

    return rezultat

def oznaci_zadetke(slika: np.ndarray, odziv: np.ndarray, prag_factor: float = 0.95) -> np.ndarray:
    # vedno vrnemo barvno sliko za prikaz
    if len(slika.shape) == 2:
        prikaz = np.stack((slika, slika, slika), axis=2).copy()
    else:
        prikaz = slika.copy()

    prag = odziv.max() * prag_factor
    maska = odziv >= prag

    # pobarvaj zadetke rdeče (BGR)
    prikaz[maska] = (0.0, 0.0, 1.0)

    return prikaz

def poisci_znak_a(slika: np.ndarray) -> np.ndarray:
    slika = slika.astype(np.float32)

    # če je barvna, jo pretvorimo v sivinsko
    if len(slika.shape) == 3:
        slika_sivinska = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
    else:
        slika_sivinska = slika.copy()

    # črne črke na belem ozadju -> obrnemo,
    # da postanejo črke svetle in ozadje temno
    slika_sivinska = 1.0 - slika_sivinska

    # filter za A
    jedro = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1]
    ], dtype=np.float32)

    rezultat = cv.filter2D(slika_sivinska, cv.CV_32F, jedro)
    rezultat = np.abs(rezultat)

    return rezultat

def oceni_orientacijo_horizonta(slika: np.ndarray) -> float:
    slika = slika.astype(np.float32)

    # 1. pretvorba v sivinsko
    if slika.ndim == 3:
        siva = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
    else:
        siva = slika.copy()

    # 2. glajenje - odstrani manjše podrobnosti
    siva = cv.GaussianBlur(siva, (5, 5), 1.0)

    # 3. Sobel gradienta
    gx = cv.Sobel(siva, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(siva, cv.CV_32F, 0, 1, ksize=3)

    # 4. moč gradienta
    moc = np.sqrt(gx * gx + gy * gy)

    # 5. smer gradienta v stopinjah
    koti_gradienta = np.degrees(np.arctan2(-gy, gx))

    # gradientne kote pretvorimo v orientacijo robov/linij
    koti_linij = koti_gradienta + 90.0

    # preslikava v interval [-90, 90]
    koti_linij = ((koti_linij + 90) % 180) - 90

    # histogram orientacij
    histogram = np.zeros(181, dtype=np.float32)   # od -90 do 90

    prag = moc.max() * 0.2
    visina, sirina = moc.shape

    for y in range(visina):
        for x in range(sirina):
            if moc[y, x] >= prag:
                kot = int(round(koti_linij[y, x]))
                kot = max(-90, min(90, kot))
                histogram[kot + 90] += moc[y, x]

    orientacija = float(np.argmax(histogram) - 90)

    # -90 in 90 za horizont obravnavamo kot 0
    if orientacija == -90.0 or orientacija == 90.0:
        orientacija = 0.0

    return orientacija


if __name__ == "__main__":
    # =========================
    # VKLOP / IZKLOP POSAMEZNIH NALOG
    # =========================
    NALOGA_1 = False
    NALOGA_2 = False
    NALOGA_3 = False
    NALOGA_4 = False
    NALOGA_5 = True

    # =========================
    # NALOGA 1 - KONVOLUCIJA
    # =========================
    if NALOGA_1:
        print("===== NALOGA 1: KONVOLUCIJA =====")

        slika1 = cv.imread(str(UTILS / "rotirani_kvadrati.png")).astype(np.float32) / 255

        # primer jedra: povprečni filter 3x3
        jedro = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ], dtype=np.float32) / 9.0

        rezultat1 = konvolucija(slika1, jedro)

        cv.imshow("Naloga 1 - Original", slika1)
        cv.imshow("Naloga 1 - Konvolucija", rezultat1)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # =========================
    # NALOGA 2 - VERTIKALNI ROBOVI (SOBEL)
    # =========================
    if NALOGA_2:
        print("===== NALOGA 2: SOBEL VERTIKALNO =====")

        slika2 = cv.imread(str(UTILS / "rotirani_kvadrati.png")).astype(np.float32) / 255

        rezultat2 = sobel_vertikalno(slika2, np.float32(0.2), (0.0, 0.0, 1.0))

        cv.imshow("Naloga 2 - Original", slika2)
        cv.imshow("Naloga 2 - Sobel vertikalno", rezultat2)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # =========================
    # NALOGA 3 - KOTICKI ROTIRANIH KVADRATOV
    # =========================
    if NALOGA_3:
        print("===== NALOGA 3: KOTICKI ROTIRANIH KVADRATOV =====")

        slika3 = cv.imread(str(UTILS / "rotirani_kvadrati.png")).astype(np.float32) / 255

        rezultat3 = poisci_koticke_rotiranih_kvadratov(slika3)

        # posamezni kanali: Z, D, L, S
        # : pomeni vzemi vse, npr vse vrstice, stolpci, tretji pa je kanal
        kanal_Z = rezultat3[:, :, 0]
        kanal_D = rezultat3[:, :, 1]
        kanal_L = rezultat3[:, :, 2]
        kanal_S = rezultat3[:, :, 3]

        # normalizacija za lepši prikaz
        kanal_Z_prikaz = (kanal_Z - kanal_Z.min()) / (kanal_Z.max() - kanal_Z.min() + 1e-8)
        kanal_D_prikaz = (kanal_D - kanal_D.min()) / (kanal_D.max() - kanal_D.min() + 1e-8)
        kanal_L_prikaz = (kanal_L - kanal_L.min()) / (kanal_L.max() - kanal_L.min() + 1e-8)
        kanal_S_prikaz = (kanal_S - kanal_S.min()) / (kanal_S.max() - kanal_S.min() + 1e-8)

        print("Max Z:", np.max(kanal_Z))
        print("Max D:", np.max(kanal_D))
        print("Max L:", np.max(kanal_L))
        print("Max S:", np.max(kanal_S))

        cv.imshow("Naloga 3 - Original", slika3)
        cv.imshow("Naloga 3 - Kanal Z", kanal_Z_prikaz)
        cv.imshow("Naloga 3 - Kanal D", kanal_D_prikaz)
        cv.imshow("Naloga 3 - Kanal L", kanal_L_prikaz)
        cv.imshow("Naloga 3 - Kanal S", kanal_S_prikaz)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # =========================
    # NALOGA 4 - DETEKCIJA CRKE A
    # =========================
    if NALOGA_4:
        print("===== NALOGA 4: DETEKCIJA CRKE A =====")

        slika4 = cv.imread(str(UTILS / "crke.png")).astype(np.float32) / 255

        odziv4 = poisci_znak_a(slika4)
        odziv4_prikaz = (odziv4 - odziv4.min()) / (odziv4.max() - odziv4.min() + 1e-8)

        # barvno označevanje zadetkov
        oznacena4 = oznaci_zadetke(slika4, odziv4, 0.95)

        # povečava za lažji prikaz, ker je slika majhna
        slika4_big = cv.resize(slika4, None, fx=6, fy=6, interpolation=cv.INTER_NEAREST)
        odziv4_big = cv.resize(odziv4_prikaz, None, fx=6, fy=6, interpolation=cv.INTER_NEAREST)
        oznacena4_big = cv.resize(oznacena4, None, fx=6, fy=6, interpolation=cv.INTER_NEAREST)

        cv.imshow("Naloga 4 - Original", slika4_big)
        cv.imshow("Naloga 4 - Odziv filtra A", odziv4_big)
        cv.imshow("Naloga 4 - Oznaceni zadetki A", oznacena4_big)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # =========================
    # NALOGA 5 - OCENA ORIENTACIJE HORIZONTA
    # =========================
    if NALOGA_5:
        print("===== NALOGA 5: ORIENTACIJA HORIZONTA =====")

        # test vseh treh slik
        imena_slik = [
            "horizont_rot_0.png",
            "horizont_rot_45.png",
            "horizont_rot_neg_45.png"
        ]

        for ime in imena_slik:
            slika5 = cv.imread(str(UTILS / ime)).astype(np.float32) / 255
            kot = oceni_orientacijo_horizonta(slika5)
            print(f"{ime} -> {kot} stopinj")

# np.zeros(shape, dtype) | shape=oblika tabele , dtype=tip podatkov , ustvari tabelo samih ničel
# np.zeros_like(a, dtype=...) | a=tabela po kateri vzame obliko , ustvari tabelo ničel enake oblike kot druga tabela
# np.stack(arrays, axis=...) | arrays/tuple več tabel , axis=po kateri osi združuje , združi več tabel v novo dimenzijo
# np.abs(x) | x=število/tabela , negativne vrednosti pretvori v pozitivne
# np.sqrt(x)
# np.degrees(x) | pretvori iz radianov v stopinje
# np.arctan2(y, x)
# np.argmax(a) | a=tabela , vrne index največje vrednosti
# np.max(a)
# np.float32(...)

# .astype(dtype) | dtype=ciljni tip , pretvori tabelo v drug data type
# .copy()
# .max()
# .min()

# cv.cvtColor(src, code) | src=vhodna slika , code = način pretvorbe , pretvori sliko iz enega barvnega prostora v drugega
# cv.COLOR_BGR2GRAY | iz barvne v sivinsko
# cv.Sobel(src, ddepth, dx, dy, ksize=...) | src=vhodna slika , ddepth=tip izhoda , dx=odvod po x , dy=odvod po y , ksize=velikost Sobel jedra
# cv.filter2D(src, ddepth, kernel) | -||- , -||- , kernel=filtrsko jedro , na sliko uporabi 2D filter
# cv.GaussianBlur(src, ksize, sigmaX) | -||- , -||- , sigmaX=standardni odklon v x smeri
# cv.resize(src, dsize, fx=..., fy=..., interpolation=...) | -||- , dsize=nova abs. velikost, če je none uporabi fx in fy (faktorja širine in višine), interpolatio=način interpolacije
# cv.INTER_NEAREST | najbližji sosed

# len(x)
# range(n)
# str(x) , int(x) , float(x)
# round(x)
# max(a, b),  min(a, b)