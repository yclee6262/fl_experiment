# 方法與流程測試

從 repo 根目錄執行 `python -m pytest -q`。

測試範圍包括 box constraints、數值 FedAvg、十法的權重組成及 cached evidence、支持訊號、paired analysis 與缺失／重複案例處理。另以 `test_alpha_calibration.py` 保留獨立校正研究的接受／拒絕、停止與結算回歸案例；本機封存的其他探索不再收集。

單元測試不需要歷史模型；用完整本機資料重播初始化、角色與成本，需依各實驗 README 另行執行。快照目錄保留的是歷史求解器，不自動當作本次 pytest 的額外測試套件。
