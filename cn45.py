import pandas as pd
import math as mth
import copy
import random
from tqdm import tqdm
import time
import mlMetrics as metrics

class BasicTreeMethod():

    def _entropy(self, positive, negative, allCount):
        # Ak je počet pozitívnych alebo negatívnych núl, entropia je nula - nie je žiadna neistota
        if positive == 0 or negative == 0:
            return 0
        # Výpočet entropie pre pozitívne a negatívne príklady, využívame vzorec pre binárnu entropiu
        entropy = (positive/allCount * mth.log2(positive/allCount) + negative/allCount * mth.log2(negative / allCount)) * -1
        return entropy

    def _conditional_entropy(self, probability, entropy):
        # Kontrola, či vstupné pole pravdepodobností a entropie majú správne rozmery a nie sú prázdne
        if len(probability) == 0 or len(probability) != len(entropy) or len(probability[0]) != 2:
            raise ValueError("[BasicTreeMethod] condition_entropy have wrong value")
        result = 0
        # Výpočet podmienenej entropie ako vážený priemer entropií
        for i in range(len(entropy)):
            result += probability[i][0] / probability[i][1] * entropy[i]
        return result

    def _info_gain(self, entropy, conditional_entropy):
        # Informačný zisk sa rovná rozdielu celkovej entropie a podmienenej entropie
        return entropy - conditional_entropy

    def _shannon_entropy(self, probability):
        # Kontrola, či vstupné pole pravdepodobnosti má správny formát
        if len(probability) == 0 or len(probability[0]) != 2:
            raise ValueError("[CN45] shannon_etropy have wrong value")
        result = 0
        # Výpočet Shannonovej entropie pre pole pravdepodobností
        for i in range(len(probability)):
            prob = probability[i][0] / probability[i][1]
            try:
                result += prob * mth.log2(prob)
            except ValueError:
                return 0
        return result * -1

    def _normalize_information_gain(self, info_gain, shannon_entropy):
        # Normalizácia informačného zisku delením Shannonovej entropie, ochrana proti deleniu nulou
        try:
            return info_gain / shannon_entropy
        except ZeroDivisionError:
            return 0

    def _find_count_pos_and_neg(self, x, y, key, attr):
        # Kontrola, či sú rozmery vstupných polí x a y rovnaké
        if len(x[key]) != len(y):
            raise ValueError("find_count_pos_and_neg")
        pos, neg, allCount = 0, 0, 0
        # Počítanie pozitívnych a negatívnych príkladov pre zadaný atribút
        for idx, item in enumerate(y):
            if x[key][idx] == attr:
                allCount += 1
                if item == "+":
                    pos += 1
                else:
                    neg += 1
        return pos, neg, allCount

    def _find_count_for_number(self, x, y, key, number, larger):
        pos = neg = allCount = 0
        # Výber príkladov, ktoré spĺňajú číselnú podmienku väčší alebo menší ako zadané číslo
        if larger:
            new_arr = [(idx, item) for idx, item in enumerate(x[key]) if float(item) > number]
        else:
            new_arr = [(idx, item) for idx, item in enumerate(x[key]) if float(item) < number]

        # Počítanie pozitívnych a negatívnych príkladov pre vyfiltrované hodnoty
        for idx, item in new_arr:
            if y[idx] == '+':
                pos += 1
            else:
                neg += 1

        return pos, neg, len(new_arr)

    def _find_pos_neg(self, y):
        pos = 0
        neg = 0
        # Počítanie počtu pozitívnych a negatívnych príkladov v poli y
        for i in y:
            if i == "+":
                pos += 1
            else:
                neg += 1
        return pos, neg, len(y)



class CN45(BasicTreeMethod):
    def __init__(self, x, y):
        # Inicializácia triedy: uloženie vstupných dát x a y
        self._x = x  # Ukladá vstupné príznaky
        self._y = y  # Ukladá cieľové hodnoty
        self._root = Node(None)  # Vytvorenie koreňového uzla stromu
        self._total_raws = len(self._y)  # Počet prvkov v y, používa sa pre progress bar

    def make_tree(self):
        # Metóda na vytvorenie rozhodovacieho stromu
        itter = 0  # Počítadlo iterácií pre debugovanie alebo analýzu
        pbar = tqdm(total=self._total_raws, desc="Building tree", unit="iter")  # Progress bar pre vizualizáciu procesu
        space_and_nodes = [[self._x, self._y, self._root]]  # Zoznam pre ukladanie súčasného stavu priestoru a uzlov
        finish = False  # Premenná určujúca, či pokračovať v cykle
        while len(space_and_nodes) != 0:  # Pokračovať kým existujú nevyriešené uzly
            new_space_and_nodes = []  # Zoznam pre nové uzly na ďalšiu iteráciu
            for x, y, node in space_and_nodes:
                total_entropy = self._entropy(*self._find_pos_neg(y))  # Vypočíta celkovú entropiu súčasného uzlu
                result = list()  # Výsledky pre každý atribút v uzle
                keys = list()  # Kľúče atribútov pre aktuálny priestor
                for key in x:
                    keys.append(key)
                    try:
                        # Kontrola, či je možné atribút spracovať ako numerický
                        int(float(x[key][0]))  # Pokus o konverziu prvej hodnoty na float, potom na int
                        result.append(self._calculate_gain_numeric(x, y, total_entropy, key))  # Výpočet zisku pre numerický atribút
                    except ValueError:
                        result.append(self._calculate_gain(x, y, total_entropy, key))  # Výpočet zisku pre ne-numerický atribút

                finish = True
                for i in self._separate_space(x, y, result, keys, node):
                    # Spracovanie každého uzla, ktorý nebol konvertovaný na list
                    if type(i[2]) != Leaf:
                        finish = False
                        new_space_and_nodes.append(i)  # Pridanie nových uzlov do zoznamu pre ďalšiu iteráciu
                        continue
                    pbar.update(len(i[1]))  # Aktualizácia progress baru
            itter += 1
            space_and_nodes = new_space_and_nodes
        pbar.close()  # Zatvorenie progress baru po dokončení

    def _separate_space(self, x, y, result, keys, parent):
        # Metóda na rozdelenie priestoru dát podľa najlepšieho zisku informácií
        best_gain = max(list(map(lambda x: x[1], result)))  # Nájdenie maximálneho zisku
        if best_gain == 0:
            plus, minus, all_count = self._find_pos_neg(y)  # Ak je zisk nula, vráti list s majoritnou triedou
            return [[None, y, Leaf('+' if plus >= minus else '-', parent)]]

        best = {}
        for idx, item in enumerate(result):
            if item[1] == best_gain:  # Nájdenie atribútov s najlepším ziskom
                best[keys[idx]] = item
        if len(best) > 1:
            best_key, [attr_value, best_score] = random.choice(list(best.items()))  # Náhodný výber pri viacerých najlepších atribútoch
        else:
            best_key, [attr_value, best_score] = list(best.items())[0]  # Výber prvku, ak je len jeden najlepší
        if attr_value is not None:
            return self._give_new_space_numeric(x, y, best_key, attr_value, parent)  # Rozdelenie priestoru pre numerické atribúty
        else:
            return self._give_new_space_attr(x, y, best_key, attr_value, parent)  # Rozdelenie priestoru pre kategorické atribúty

    def make_prediction(self, x):
        # Metóda pre predikciu klasifikačného výsledku pre vstup x
        current_pos = self._root  # Začíname od koreňového uzla
        finish = False  # Premenná na kontrolu ukončenia cyklu
        while not finish:
            attrName = current_pos.attr_name  # Meno atribútu aktuálneho uzla
            attrValue = current_pos.attr_value  # Hodnota atribútu aktuálneho uzla
            new_pos = None  # Premenná pre novú pozíciu v strome
            if attrValue is not None:
                # Porovnanie hodnoty atribútu s hodnotou v uzle
                if float(x[current_pos.attr_name]) > float(current_pos.attr_value):
                    new_pos = current_pos.get_childs()[0][0]  # Priradenie ľavého dieťaťa
                else:
                    new_pos = current_pos.get_childs()[1][0]  # Priradenie pravého dieťaťa
            else:
                for i in current_pos.get_childs():
                    try:
                        # Prechádzanie kategorických atribútov
                        if x[attrName] == i[1]:
                            new_pos = i[0]  # Nastavenie nového uzla na základe zhody
                    except KeyError:
                        return i[0].ClassName  # Vrátenie mena triedy, ak dôjde k chybe
            current_pos = new_pos  # Aktualizácia aktuálnej pozície
            if type(current_pos) is Leaf:
                finish = True  # Ukončenie, ak je nový uzol list
            elif current_pos is None:
                return random.choice(['+', '-'])  # Náhodný výber, ak uzol neexistuje
        return current_pos.ClassName  # Vrátenie mena triedy listu

    def _get_root_node(self):
        # Vráti koreňový uzol stromu
        return self._root

    def _give_new_space_numeric(self, x, y, best_key, attr_value, parent):
        # Rozdelenie priestoru pre numerické atribúty
        result = [{i: [] for i in x} for _ in range(2)]  # Inicializácia zoznamu pre rozdelené dáta
        y_new = [[] for _ in range(2)]  # Inicializácia zoznamu pre rozdelené ciele
        for idx, item in enumerate(x[best_key]):
            if float(item) > attr_value:
                y_new[0].append(y[idx])  # Priradenie do prvej skupiny
                for i in x:
                    result[0][i].append(x[i][idx])  # Priradenie príznakov do prvej skupiny
            else:
                y_new[1].append(y[idx])  # Priradenie do druhej skupiny
                for i in x:
                    result[1][i].append(x[i][idx])  # Priradenie príznakov do druhej skupiny
        for_return = []
        parent.set_attr_name(best_key)  # Nastavenie názvu atribútu pre rodičovský uzol
        parent.set_attr_value(attr_value)  # Nastavenie hodnoty atribútu pre rodičovský uzol
        for idx, item in enumerate(y_new):
            for key in copy.deepcopy(result[idx]):
                if len(set(result[idx][key])) <= 1:
                    result[idx].pop(key)  # Odstránenie neefektívnych atribútov
            if self._entropy(*self._find_pos_neg(item)) == 0:
                for_return.append([None, item, Leaf(item[0], parent)])  # Pridanie listu, ak entropia je nula
            elif len(result[idx]) == 0 or len(item) < 3:
                plus, minus, all_count = self._find_pos_neg(item)
                for_return.append([None, item, Leaf('+' if plus >= minus else '-', parent)])  # Pridanie listu na základe väčšiny
            else:
                for_return.append([result[idx], item, Node(parent)])  # Pridanie nového uzla
        return for_return

    def _give_new_space_attr(self, x, y, best_key, attr_value, parent):
        # Rozdelenie priestoru pre kategorické atribúty
        set_attributes = set([i for i in x[best_key]])  # Získanie unikátnych hodnôt atribútu
        result = {_:{i: [] for i in x if i != best_key or len(set(x[i])) > 1} for _ in list(set_attributes)}
        y_new = {_: [] for _ in list(set_attributes)}  # Inicializácia zoznamov pre ciele
        for idx, item in enumerate(x[best_key]):
            y_new[item].append(y[idx])  # Rozdelenie cieľov podľa hodnoty atribútu
            for i in x:
                if i != best_key or len(set(x[i])) > 1:
                    result[item][i].append(x[i][idx])  # Rozdelenie príznakov, okrem aktuálneho atribútu
        for_return = []
        parent.set_attr_name(best_key)  # Nastavenie názvu atribútu pre rodičovský uzol
        parent.set_attr_value(attr_value)  # Nastavenie hodnoty atribútu pre rodičovský uzol
        for i in result:
            for key in copy.deepcopy(result[i]):
                if len(set(result[i][key])) <= 1:
                    result[i].pop(key)  # Odstránenie neefektívnych atribútov
            if self._entropy(*self._find_pos_neg(y_new[i])) == 0:
                for_return.append([None, y_new[i], Leaf(y_new[i][0], parent, i)])  # Pridanie listu, ak entropia je nula
            elif len(result[i]) == 0 or len(y_new[i]) <= 3:
                plus, minus, all_count = self._find_pos_neg(y_new[i])
                for_return.append([None, y_new[i], Leaf('+' if plus >= minus else '-', parent, i)])  # Pridanie listu na základe väčšiny
            else:
                for_return.append([result[i], y_new[i], Node(parent, child_attr_value=i)])  # Pridanie nového uzla
        return for_return



    def _calculate_gain_numeric(self, x, y, total_entropy, attr):
        # Získanie unikátnych hodnôt z atribútu a ich usporiadanie
        #numb_set = sorted([i for i in set([round(item) for item in x[attr]])])
        numb_set = sorted([i for i in set(x[attr])])
        # Vypočítanie stredných hodnôt medzi každými dvoma susednými hodnotami v zozname
        numb_set = [(float(numb_set[idx]) + float(numb_set[idx + 1])) / 2 for idx in range(len(numb_set) - 1)]
        best = [0, 0]  # Inicializácia pre najlepší zisk a prahovú hodnotu
        for key in numb_set:
            vals = []
            # Získanie počtu pozitívnych a negatívnych príkladov nad a pod prahovou hodnotou
            vals.append(self._find_count_for_number(x, y, attr, key, True))
            vals.append(self._find_count_for_number(x, y, attr, key, False))
            entropies = []
            probabilities = []
            # Výpočet entropií a pravdepodobností pre každú skupinu
            for val in vals:
                entropies.append(self._entropy(*val))
                probabilities.append((val[2], len(y)))
            # Výpočet podmienenej entropie a informačného zisku
            condition_entr = self._conditional_entropy(probabilities, entropies)
            info_gain = self._info_gain(total_entropy, condition_entr)
            # Výpočet Shannonovej entropie a normalizovaného informačného zisku
            shannon = self._shannon_entropy(probabilities)
            norm_gain = self._normalize_information_gain(info_gain, shannon)
            # Uloženie najlepších výsledkov
            if norm_gain > best[1]:
                best[1] = norm_gain
                best[0] = key
        return best



    def _calculate_gain(self, x, y, total_entropy, attr):
        attr_entropy = []
        probabilities = []
        # Pre každý unikátny kľúč v atribúte
        for key in set(x[attr]):
            # Získanie počtu pozitívnych a negatívnych príkladov pre daný kľúč
            pos, neg, allCount = self._find_count_pos_and_neg(x, y, attr, key)
            # Výpočet entropie pre daný kľúč
            attr_entropy.append(self._entropy(pos, neg, allCount))
            # Vytvorenie pravdepodobností pre podmienenú entropiu
            probabilities.append((allCount, len(y)))
        # Výpočet podmienenej entropie a informačného zisku pre kategorické atribúty
        condition_entr = self._conditional_entropy(probabilities, attr_entropy)
        info_gain = self._info_gain(total_entropy, condition_entr)
        # Výpočet Shannonovej entropie a normalizovaného informačného zisku
        shannon = self._shannon_entropy(probabilities)
        norm_gain = self._normalize_information_gain(info_gain, shannon)
        return [None, norm_gain]


class ReadCsv():
    def __init__(self, file_path, className, label):
        # Výpis o načítaní CSV súboru
        print("ReadCsv")
        # Načítanie CSV súboru a náhodné premiešanie riadkov
        data = pd.read_csv(file_path).sample(frac=1)
        # Konverzia DataFrame do slovníka
        data_dict = data.to_dict()
        self._dictData = dict()
        # Prechod cez všetky kľúče v slovníku a transformácia údajov do zoznamu
        for i in data_dict:
            self._dictData[i] = [j[1] for j in data_dict[i].items()]
        # Kopírovanie údajov pre ďalšie spracovanie
        data_dict = copy.deepcopy(self._dictData)
        self._y = []
        # Rozdelenie údajov na vstupné (x) a výstupné (y) na základe className
        for i in data_dict.pop(className):
            # Priradenie '+' alebo '-' v závislosti od zhody s cieľovou triedou
            if str(i) == label:
                self._y.append('+')
            else:
                self._y.append('-')
        self._x = data_dict  # Uloženie vstupných príznakov

    def get_x(self):
        # Vrátenie vstupných údajov
        return self._x

    def get_y(self):
        # Vrátenie cieľových údajov
        return self._y

    def _prepare_data(self, x, y):
        # Vytvorenie progress baru pre vizualizáciu prípravy údajov
        pbar = tqdm(total=len(y)+1, desc="Prepare data", unit="iter")
        similarity = []
        # Hľadanie duplicitných alebo konfliktných záznamov
        for idx, y_item in enumerate(y):
            main_item = {i: x[i][idx] for i in x}
            for idx_2 in range(idx + 1, len(y)):
                some_dict = {i: x[i][idx_2] for i in x}
                # Porovnanie, či sú záznamy identické
                if some_dict == main_item:
                    similarity.append(idx)
            pbar.update(1)
        # Odstránenie duplicitných záznamov
        similarity = list(set(similarity))
        new_x, new_y = {key: [] for key in x}, []
        for idx in range(len(y)):
            if idx in similarity:
                continue
            for key in x:
                new_x[key].append(x[key][idx])
            new_y.append(y[idx])
        pbar.update(1)
        pbar.close()
        # Vrátenie upravených údajov
        return new_x, new_y

    def returnDataSplitter(self):
        # Vrátenie inštancie DataSplitter pre ďalšie rozdelenie údajov
        return DataSplitter(self._x, self._y)




class DataSplitter():
    def __init__(self, x, y):
        # Inicializácia triedy so vstupnými dátami x a cieľovými hodnotami y
        self._x = x  # Ukladá vstupné príznaky
        self._y = y  # Ukladá cieľové hodnoty

    def split_data(self, train_per):
        # Metóda na rozdelenie dát na tréningovú a testovaciu množinu podľa percentuálneho zastúpenia
        train_length = int(len(self._y) / 100 * train_per)  # Vypočítanie počtu prvkov v tréningovej množine
        print({i for i in self._x})  # Výpis množiny názvov stĺpcov
        train_x = dict()  # Slovník pre tréningové príznaky
        test_x = dict()  # Slovník pre testovacie príznaky
        train_y = self._y[:train_length]  # Získanie tréningových cieľových hodnôt
        test_y = self._y[train_length:]  # Získanie testovacích cieľových hodnôt
        for i in self._x:
            train_x[i] = self._x[i][:train_length]  # Získanie tréningových príznakov pre každý stĺpec
            test_x[i] = self._x[i][train_length:]  # Získanie testovacích príznakov pre každý stĺpec

        return [train_x, train_y], [test_x, test_y]  # Vrátenie rozdelených dát

    def make_portion(self, portions_count):
        # Metóda na rozdelenie dát na viacero častí
        portions = []  # Zoznam na ukladanie jednotlivých častí
        portion_size = len(self._y) // portions_count  # Vypočítanie veľkosti jednej časti
        start = 0  # Začiatok aktuálnej časti
        end = portion_size  # Koniec aktuálnej časti
        for i in range(portions_count):  # Pre každú časť
            new_x = dict()  # Slovník pre príznaky aktuálnej časti
            for i in self._x:
                new_x[i] = self._x[i][start:end]  # Získanie príznakov pre časť
            portions.append([new_x, self._y[start:end]])  # Pridanie časti do zoznamu
            start = end  # Aktualizácia začiatku na koniec predchádzajúcej časti
            end += portion_size  # Posunutie konca na ďalšiu časť
        for i in self._y[start:]:  # Pre zvyšné hodnoty, ktoré nezapadajú presne do častí
            for x, y in portions:
                for key in self._x:
                    x[key].append(self._x[key][start])  # Pridanie hodnôt príznakov do častí
                y.append(i)  # Pridanie cieľových hodnôt do častí
            start += 1  # Posunutie indexu na ďalšiu hodnotu
        return portions  # Vrátenie rozdelených častí



class Node():
    def __init__(self, parent, attr_value=None, child_attr_value=None):
        # Inicializácia uzla s rodičom a prípadnými hodnotami atribútov
        self._parent = parent  # Ukladanie referencie na rodičovský uzol
        # Ak uzol má rodiča, pridá sa tento uzol do zoznamu detí rodiča
        if self._parent is not None:
            self._parent.add_child(self, child_attr_value)
        self._childs = []  # Inicializácia zoznamu detí
        self._attr_name = None  # Atribút meno bude nastavené neskôr
        self._attr_value = None  # Atribút hodnota bude nastavené neskôr

    def set_attr_name(self, attr_name):
        # Nastavenie mena atribútu pre uzol
        self._attr_name = attr_name

    def set_attr_value(self, attr_value):
        # Nastavenie hodnoty atribútu pre uzol
        self._attr_value = attr_value

    def add_child(self, child, attr_value=None):
        # Pridanie dieťaťa do zoznamu detí s prípadnou hodnotou atribútu
        self._childs.append([child, attr_value])

    @property
    def attr_value(self):
        # Vráti hodnotu atribútu uzla
        return self._attr_value

    @property
    def attr_name(self):
        # Vráti meno atribútu uzla
        return self._attr_name

    def get_childs(self):
        # Vráti zoznam detí uzla
        return self._childs

class Leaf():
    def __init__(self, className, parent, parent_attr_value=None):
        # Inicializácia listu s menom triedy a rodičom
        self._className = className  # Ukladanie mena triedy pre list
        self._parent = parent  # Ukladanie referencie na rodičovský uzol
        # Pridanie tohto listu do zoznamu detí rodiča s prípadnou hodnotou atribútu
        self._parent.add_child(self, parent_attr_value)

    @property
    def ClassName(self):
        # Vráti meno triedy listu
        return self._className




class BaggingCN45():
    def __init__(self, x, y, tree_count, portion_count, boot_strap):
        # Inicializácia triedy pre bagging s rozhodovacími stromami CN45
        self._dataSplitter = DataSplitter(x, y)  # Vytvorenie objektu pre rozdelenie dát
        self._portions = self._dataSplitter.make_portion(portion_count)  # Vytvorenie častí dát na základe počtu častí
        self._trees = []  # Zoznam na ukladanie stromov
        # Vytvorenie stromov na základe bootstrappingu
        for i in range(tree_count):
            portion = [{key: [] for key in x}, []]  # Vytvorenie prázdných dát pre každý strom
            for _ in range(boot_strap):
                some = random.choice(self._portions)  # Náhodný výber jednej časti
                for key in some[0]:
                    portion[0][key] += some[0][key]  # Zlučovanie dát pre strom
                portion[1] += some[1]  # Zlučovanie cieľových hodnôt
            self._trees.append(CN45(portion[0], portion[1]))  # Vytvorenie stromu CN45
        for i in self._trees:
            i.make_tree()  # Vytvorenie stromu z dát

    def make_prediction(self, test_x, test_y, print_accuracy=False):
        # Metóda na predikciu a voliteľné výpočet presnosti
        randomKey = random.choice([i for i in test_x])  # Náhodný výber kľúča z testovacích dát
        prediction = []  # Zoznam na ukladanie výsledkov predikcií
        for idx, item in enumerate(test_x[randomKey]):
            x_1 = dict()
            for _ in test_x:
                x_1[_] = test_x[_][idx]  # Vytvorenie dátového bodu pre predikciu
            y_predicts = []
            for tree in self._trees:
                y_predicts.append(tree.make_prediction(x_1))  # Získanie predikcií od všetkých stromov
            if y_predicts.count('+') > y_predicts.count('-'):
                prediction.append('+' == test_y[idx])  # Rozhodnutie, či väčšina stromov vrátila '+'
            else:
                prediction.append('-' == test_y[idx])  # Rozhodnutie, či väčšina stromov vrátila '-'

        if print_accuracy:
            print(len(test_y))
            accuracy = sum(prediction) / len(test_y)  # Výpočet presnosti
            print(accuracy)
            return accuracy

    def return_prediction(self, test_x, test_y):
        # Metóda na vrátenie štatistiky predikcií
        randomKey = random.choice([i for i in test_x])
        tp = 0  # True positive
        fp = 0  # False positive
        tn = 0  # True negative
        fn = 0  # False negative
        for idx, item in enumerate(test_x[randomKey]):
            x_1 = dict()
            for _ in test_x:
                x_1[_] = test_x[_][idx]  # Vytvorenie dátového bodu pre predikciu
            y_predicts = []
            for tree in self._trees:
                y_predicts.append(tree.make_prediction(x_1))  # Získanie predikcií od všetkých stromov
            if y_predicts.count('+') > y_predicts.count('-'):
                if '+' == test_y[idx]:
                    tp += 1
                else:
                    fp += 1
            else:
                if '-' == test_y[idx]:
                    tn += 1
                else:
                    fn += 1

        return tp, fp, tn, fn  # Vrátenie štatistiky predikcií
