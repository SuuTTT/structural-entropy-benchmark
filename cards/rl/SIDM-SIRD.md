# Repro card — SIDM / SIRD (structural-information decision making; role discovery)

**Paper:** Zeng et al., SIDM, JMLR 2025 (omnibus: SIRD role discovery + SISA
state abstraction + SISL skill learning).
**Upstream repo:** https://github.com/SELGroup/SIDM
**User's working reproduction (USE THIS):** https://github.com/SuuTTT/reproduce-SIDM
(context.md, SIRD-context.md, SISA-context.md, SISL-context.md, launch.md, experiment.md)

## Tractability: HEAVY — do after SI2E
SIRD needs **StarCraft II v4.10 (~5GB, 24k+ files) + SMAC + MuJoCo 2.1**.
Disk-hungry; borrowed A4000s have only 9–14GB free → needs a dedicated/rented box
with ≥30GB. Reproduce SI2E (MiniGrid) first; treat SIRD as a stretch case study.

## Env (from user's context.md)
Python 3.12; PyYAML, sacred, pysc2, smac, tensorboardX, scikit-learn.
**Py3.12 fixes:** `collections.Mapping`→`collections.abc.Mapping`;
`yaml.load()`→add `Loader=yaml.FullLoader`; `np.float`→`float`; protobuf→3.20.0.

## StarCraft II setup
```bash
unzip -o -P iagreetotheeula SC2.4.10.zip   # verify 24k+ files (SC2Data/data/data.0xx)
mkdir -p Maps/SMAC_Maps && mv *.SC2Map Maps/SMAC_Maps/
export SC2PATH=".../StarCraftII"; export LD_LIBRARY_PATH="$SC2PATH/Libs:$LD_LIBRARY_PATH"
```

## Run (SIRD, role discovery)
```bash
cd SIRD/src
python main.py --config=rode --env-config=sc2 with env_args.map_name=6h_vs_8z t_max=5050000
# batch: ./run_all_sird_maps.sh  (23 SMAC maps)
```
Metrics via sacred (`results/sacred/.../run.json`) + TB logs (win rate per map).

## Gotchas (user)
Map discovery → manual place in Maps/SMAC_Maps/; role-discovery crash ~40–50k
steps = numpy deprecation; incomplete SC2 extraction → ConnectError (re-extract `-o`).
SISA/SISL partially configured.

## Plan
Phase-3 stretch: reproduce SIRD win-rate on 1–2 representative SMAC maps
(e.g. 3m easy + 6h_vs_8z hard) vs the RODE/QMIX baselines the repo ships, ≥3 seeds.
t_max=5.05M steps is long (hours–day/seed) → scope tightly; report honestly if
only partial. SISA (state abstraction) may be a lighter alternative within SIDM.

## Results
Pending (Phase 3, after SI2E).
