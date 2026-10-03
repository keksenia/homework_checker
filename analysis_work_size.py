import csv, json, random
from collections import defaultdict
for run in ["low","low_key"]:
    rows=[r for r in csv.DictReader(open(f"reports/google_gemini-3.8-flash@{run}_tasks.csv",encoding="utf-8")) if r["label"] in("0","1") and r["split"]=="dev"]
    byw=defaultdict(list)
    for r in rows: byw[r["work"]].append(r)
    # позиция задачи в ответе модели
    pos={}
    for w in byw:
        res=json.load(open(f"runs/google_gemini-3.8-flash@{run}/{w}/check.json",encoding="utf-8"))["results"]
        order=[x["task_no"] for x in res]
        for r in byw[w]:
            pos[(w,r["task_no"])]=(order.index(r["task_no"])/max(len(order)-1,1)) if r["task_no"] in order else None
    miss=lambda rs: sum(r["pred"]!=r["label"] for r in rs)
    print(f"\n@{run}")
    bins=[(1,9),(10,19),(20,34),(35,99)]
    for a,b in bins:
        ws=[w for w in byw if a<=len(byw[w])<=b]; rs=[r for w in ws for r in byw[w]]
        if rs: print(f"  работы с {a}-{b} задачами: работ {len(ws)}, задач {len(rs)}, промахов {miss(rs)} ({miss(rs)/len(rs):.1%}), нет в ответе {sum(r['verdict']=='нет в ответе' for r in rs)}")
    big=[w for w in byw if len(byw[w])>=20]
    for name,cond in [("первая половина",lambda p:p<0.5),("вторая половина",lambda p:p>=0.5)]:
        rs=[r for w in big for r in byw[w] if pos[(w,r['task_no'])] is not None and cond(pos[(w,r['task_no'])])]
        print(f"  большие работы (≥20), {name} списка: задач {len(rs)}, промахов {miss(rs)} ({miss(rs)/len(rs):.1%})")
    nopos = sum(1 for w in big for r in byw[w] if pos[(w, r['task_no'])] is None)
    if nopos:
        print(f"  (у {nopos} задач больших работ номер в ответе модели записан иначе — позиция не определена)")
    # Спирмен по долям промахов в работах смещён: у маленькой работы доля часто ровно 0 (0 промахов из 3 задач),
    # у большой почти всегда есть хоть один промах — корреляция появляется «из дискретности», а не из усталости.
    # Поэтому сравниваем объединённые доли промахов «маленькие (<20 задач) vs большие (≥20)» на уровне задач,
    # а интервал разницы — bootstrap по работам.
    small = [w for w in byw if len(byw[w]) < 20]; large = [w for w in byw if len(byw[w]) >= 20]
    rate = lambda ws: sum(miss(byw[w]) for w in ws) / sum(len(byw[w]) for w in ws)
    diffs = []
    rnd = random.Random(0)
    for _ in range(2000):
        s_ = [rnd.choice(small) for _ in small]; l_ = [rnd.choice(large) for _ in large]
        diffs.append(rate(l_) - rate(s_))
    diffs.sort()
    print(f"  доля промахов: маленькие работы {rate(small):.1%}, большие {rate(large):.1%}; "
          f"разница «большие − маленькие» {100*(rate(large)-rate(small)):+.1f} п.п. "
          f"[{100*diffs[50]:+.1f}; {100*diffs[1949]:+.1f}] (bootstrap по работам)")
