# 貢獻與影子付款的起點穩定度

依[protocol.json](protocol.json)，固定20個歷史模型團隊、成員與均勻alpha，T=±.5，交叉5組本地反推起點×3組custom求解起點，共600任務。原資料與舊輸出不修改；不重新招募、篩選或訓練。高反推誤差及失敗保留，不另抽案例。

每組本地起點產生候選後，三個求解起點為候選平均與兩個固定Dirichlet凸組合。組合係數跨反推與目標共用；換反推會同時改變候選與實際暖起點，這是反推初始化的流程效果，不能稱為只改候選而起點座標不變。換求解起點時，候選／模型／目標固定，且仍在原候選仿射空間內。

逐任務重算完整軌跡、CA／CM／J及共同起點whole-task LOO。符號tol1e-10與錨點付款門檻tauA分開；F與Fminus各套原100基本費＋500錨點池＋500模型池影子規則及整數largest remainder，保留reserve與基本費取消。沒有實際付款，主政策留任務12。

每隊先平均固定另一因子的全配對差異：反推軸30對、求解軸15對，再於每情境10隊做200000次bootstrap。報符號／正負翻轉、資格改變、含reserve的分配TV與金額變動；隊內起點與owner不當成獨立樣本，不自行設定穩定通過門檻。零事件退化區間不代表族群零風險。

另外對r0w0的40個任務新建完全匹配的子聯盟賽局：共同起點、共同完整F_N評分panel、空聯盟保留起點不搜尋；移除成員的模型與候選、剩餘alpha重正規化。精確Shapley與whole-task LOO使用同一效用；真值效用另作離線參照。共同起點保留full候選資訊，固定評分仍使用被移除模型，這是條件式反事實，不是獨立聯盟、部署付款或公平真值。Fminus每人panel不同，不拼接為同一Shapley賽局。

歷史600個三人賽局另依[SHAPLEY.md](SHAPLEY.md)核對來源與效用，不能直接和這20隊的局部角色分數配對。

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python run.py start-stability \
  --output-dir outputs/contribution_start_stability/new-run
OMP_NUM_THREADS=1 python -m experiments.contribution.start_stability.validate \
  --source outputs/contribution_start_stability/new-run \
  --reference outputs/nonzero_role_targets/task9_final_v1
OPENBLAS_NUM_THREADS=1 python -m experiments.contribution.start_stability.analyze \
  --source outputs/contribution_start_stability/new-run --output-dir outputs/contribution_start_stability/new-run/analysis
OPENBLAS_NUM_THREADS=1 python -m experiments.contribution.start_stability.validate_statistics \
  --source outputs/contribution_start_stability/new-run
```

輸出拒絕覆寫；`--smoke`以第一隊T+.5執行2×2組合，仍用正式500步反推及30輪求解，不作正式統計。原20隊模型與cases須另行提供。manifest在執行前鎖實際求解函式、設計與模型來源；`--reference`可省略，提供時另外逐筆驗證r0w0與任務九的40個基準及完整事件一致。

`preparations.json`列200個實際反推準備，每個共享三個warm起點；`index.json`列600任務。各任務cache獨立，總cache misses不是各角色方法獨立冷快取成本；observer等價檢查另列，matched Shapley只加新增misses。統計及來源重播不新增模型query，原模型驗算的計算另記。

已驗收結果與完整數據欄位見[results](results/README.md)。`python -m experiments.contribution.start_stability.plot --analysis RUN/analysis`輸出全部16個付款TV條件的描述圖；原始模型回覆、圖及逐案紀錄留於outputs。
