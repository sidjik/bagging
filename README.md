##### Priečinok obsahuje súbory, ktoré obsahujú implementáciu algoritmu, konkrétne súbory - > cn45.py, mlMetrics.py

##### Obsahuje súbor test.py, ktorý sa používal na zostavenie rôznych grafov pri písaní dokumentácie, ale je to chaos, keďže veľa grafov bolo vytvorených z jedného súboru.

### Súbor for_cvic.py je súbor na spustenie algoritmu, ktorý zobrazuje jeho metriky. Vyrobené špeciálne pre obhajobu v 13 týždňe.

#### Pozor: Program pri volaní používa nasledujúce parametre:
##### -f, --table_name: názov súboru tabuľky
##### -y: názov stĺpca, ktorý označuje cieľový atribút
##### -l, --label: ako je označená pozitívna trieda v cieľovom atribúte
##### --treeCount: počet použitých stromov pre Bagging
##### -p, --portion_count: počet kúskov, do ktorých budú rozdelené tréningové dáta
##### -b, --boot_strap: počet kusov, ktoré sa použijú na stavbu jedného stromu


## Príklady použitia s údajmi, ktoré sú v priečinku: 
 
### python for_cvic.py -f Employee.csv -y LeaveOrNot -l 1 -p 12 -b 5 --treeCount 10

### python for_cvic.py -f apple_quality.csv -y Quality -l good -p 21 -b 7 --treeCount 12

### python for_cvic.py -f startupdata.csv -y status -l closed -p 12 -b 5 --treeCount 7 
