# 歷史匯入相容支援

保留模組仍使用部分平面名稱，例如 `host_server`、`initial_alpha_methods`。`legacy_modules.json` 將這些名稱對應到分類後的正式原始碼。`prepare()` 在忽略的 `.local_scripts/` 產生相對連結並設定 Python 搜尋路徑。

只有這份精簡映射中的模組會註冊；移入 `.local_archive/` 的舊實驗不再自動匯入。`run.py`、core 匯入與 pytest 會準備需要的相容視圖。分享 repo 不需要一起複製 `.local_scripts/`。
