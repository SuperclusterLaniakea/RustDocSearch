#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// ========== 引入依赖 ==========
use anyhow::{Context, Result};                     // 错误处理
use eframe::egui::{self, Align, Color32, Layout, RichText, ScrollArea, Sense, Vec2};
use jieba_rs::Jieba;                              // 中文分词
use notify::{Event, EventKind, RecursiveMode, Watcher}; // 文件系统监控
use rfd::FileDialog;                              // 原生文件夹/文件选择对话框
use std::collections::{HashMap, VecDeque};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender}; // 多线程消息传递
use std::sync::{Arc, Mutex};
use std::time::UNIX_EPOCH;
use tantivy::collector::TopDocs;                  // 检索结果收集器
use tantivy::query::QueryParser;                  // 查询解析
use tantivy::schema::*;                           // 索引模式
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument};
use walkdir::WalkDir;                             // 递归遍历目录

// ---------- 数据结构 ----------
/// 文件元数据，存储在 sled 中，用于对比文件是否变动
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct DocMeta {
    path: PathBuf,      // 始终为绝对路径
    filename: String,   // 文件名
    mtime: u64,         // 修改时间（UNIX 时间戳）
    md5: String,        // 文件内容 + 长度 的 MD5（用于快速对比）
}

/// 单个关键词命中信息
#[derive(Clone, Debug)]
struct KeywordHit {
    location: u32,      // 在文档中的位置（页码/段落号）
    snippet: String,    // 上下文片段
    keyword: String,    // 命中的关键词
}

/// 一条搜索结果，对应一个文档
#[derive(Clone, Debug)]
struct SearchResult {
    file_id: u64,       // 内部文件 ID
    filename: String,   // 文件名
    path: PathBuf,      // 绝对路径
    location: u64,      // 命中位置
    snippet: String,    // 摘要文本（用于结果列表）
    score: f32,         // 相关度评分
    mtime: u64,         // 修改时间
    keyword_count: usize, // 该文档中所有关键词的总出现次数
    hits: Vec<KeywordHit>, // 所有命中详情
}

/// 排序方式
#[derive(Debug, Clone, PartialEq)]
enum SortBy {
    Relevance,      // 相关度
    FileNameAsc,    // 文件名 A→Z
    FileNameDesc,   // 文件名 Z→A
    DateNewest,     // 最新修改
    DateOldest,     // 最早修改
    Frequency,      // 词频
}

/// 用户配置，可持久化到 JSON
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Config {
    index_dir: String,              // 索引数据存放目录
    last_root: Option<String>,      // 上次选择的文件夹路径
    pdf_reader: Option<String>, 
    word_reader: Option<String>,     // PDF 阅读器命令行模板
}

impl Default for Config {
    fn default() -> Self {
        let base = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".doc_searcher");
        Self {
            index_dir: base.to_string_lossy().to_string(),
            last_root: None,
            pdf_reader: None,
            word_reader: None,
        }
    }
}

/// 索引器状态
#[derive(Debug, Clone, PartialEq)]
enum IndexState {
    Idle,
    Indexing,
    Paused,
}

/// 索引线程发出的消息
enum IndexMsg {
    Progress {
        current: String, // 当前正在处理的文件路径
        total: usize,    // 总文件数
        index: usize,    // 当前序号
    },
    Done,
}

// ---------- 安全切片辅助函数 ----------
/// 安全的 UTF-8 切片，确保切分在字符边界处，避免 panic
fn safe_slice(text: &str, start: usize, len: usize) -> &str {
    let mut real_start = start;
    while real_start > 0 && !text.is_char_boundary(real_start) {
        real_start -= 1;
    }
    let end = (real_start + len).min(text.len());
    let mut real_end = end;
    while real_end < text.len() && !text.is_char_boundary(real_end) {
        real_end += 1;
    }
    &text[real_start..real_end]
}

// ---------- 分词 ----------
/// 对文本进行分词，返回空格分隔的词序列
fn tokenize(text: &str) -> String {
    let jieba = Jieba::new();
    jieba.cut(text, true).join(" ")
}

/// 对查询词分词
fn tokenize_query(query: &str) -> String {
    tokenize(query)
}

// ---------- 文件 MD5 ----------
/// 计算文件的快速 MD5（内容首块 + 文件长度），用于检测文件变化
fn compute_md5(path: &Path) -> Result<String> {
    let mut hasher = md5::Context::new();
    let mut file = std::fs::File::open(path)?;
    use std::io::Read;
    let mut buffer = [0u8; 8192];
    let n = file.read(&mut buffer)?;
    hasher.consume(&buffer[..n]);
    hasher.consume(&format!("{}", file.metadata()?.len()));
    Ok(format!("{:x}", hasher.compute()))
}

// ---------- 文本提取 ----------
/// 带 panic 保护的文本提取，避免外部库崩溃导致主程序退出
fn extract_pages_safe(path: &Path, ext: &str) -> Result<Vec<(u32, String)>> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        extract_pages(path, ext)
    }));
    match result {
        Ok(res) => res,
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "unknown panic".to_string()
            };
            Err(anyhow::anyhow!("extract_pages panicked: {}", msg))
        }
    }
}

/// 根据文件扩展名提取文本，返回 (页码/段落号, 文本内容) 的列表
fn extract_pages(path: &Path, ext: &str) -> Result<Vec<(u32, String)>> {
    match ext {
        "pdf" => {
            let bytes = std::fs::read(path)?;
            let text = pdf_extract::extract_text_from_mem(&bytes)
                .map_err(|e| anyhow::anyhow!("PDF extract error: {}", e))?;
            Ok(vec![(1, text)]) // 目前整个 PDF 视为一页
        }
        "docx" => {
            let data = std::fs::read(path)?;
            let docx = docx_rs::read_docx(&data)
                .map_err(|e| anyhow::anyhow!("DOCX error: {}", e))?;
            let mut paras = Vec::new();
            for child in &docx.document.children {
                if let docx_rs::DocumentChild::Paragraph(p) = child {
                    let mut para_text = String::new();
                    for pchild in &p.children {
                        if let docx_rs::ParagraphChild::Run(run) = pchild {
                            for rchild in &run.children {
                                if let docx_rs::RunChild::Text(t) = rchild {
                                    para_text.push_str(&t.text);
                                }
                            }
                        }
                    }
                    if !para_text.trim().is_empty() {
                        paras.push(para_text);
                    }
                }
            }
            Ok(paras
                .into_iter()
                .enumerate()
                .map(|(i, t)| (i as u32 + 1, t))
                .collect())
        }
        "txt" => {
            let text = std::fs::read_to_string(path)?;
            Ok(vec![(1, text)])
        }
        _ => Ok(vec![]),
    }
}

// ---------- 应用主状态 ----------
struct DocSearcherApp {
    // tantivy 相关
    index: Index,
    reader: IndexReader,
    writer: Arc<Mutex<IndexWriter>>,
    schema: Schema,
    body_field: Field,        // 文本字段
    file_id_field: Field,     // 文件 ID 字段
    location_field: Field,    // 位置字段

    meta_db: sled::Db,        // 文件元数据数据库

    config: Config,           // 用户配置
    config_path: PathBuf,     // 配置文件路径

    root_dir: Option<PathBuf>, // 当前索引根目录（绝对路径）
    index_state: IndexState,   // 索引状态
    pause_flag: Arc<AtomicBool>, // 暂停标志
    stop_flag: Arc<AtomicBool>,  // 停止标志
    index_status: String,       // 状态信息显示

    log_messages: VecDeque<String>, // 日志消息
    total_indexed: usize,           // 已索引文件数
    current_processing: String,     // 当前正在处理的文件

    compare_result: Option<(usize, usize, usize)>, // 对比结果：新增、修改、删除

    // 搜索相关
    search_query: String,
    results: Vec<SearchResult>,
    sort_by: SortBy,
    selected_result: Option<usize>,   // 选中的搜索结果索引
    selected_hit: Option<usize>,      // 选中的命中索引

    // 设置界面
    show_settings: bool,
    temp_index_dir: String,
    temp_pdf_reader: String,

    // 文件监控
    _watcher: Option<notify::RecommendedWatcher>,
    _watcher_handle: Option<std::thread::JoinHandle<()>>,
    progress_rx: Option<Receiver<IndexMsg>>, // 接收索引进度

    pending_open: Option<SearchResult>, // 待打开的文件
}

impl DocSearcherApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // 确保配置目录存在
        let config_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".doc_searcher");
        std::fs::create_dir_all(&config_dir).ok();
        let config_path = config_dir.join("config.json");
        let config = if config_path.exists() {
            std::fs::read_to_string(&config_path)
                .ok()
                .and_then(|s| serde_json::from_str::<Config>(&s).ok())
                .unwrap_or_default()
        } else {
            Config::default()
        };

        // 创建索引及元数据目录
        let index_dir = PathBuf::from(&config.index_dir);
        std::fs::create_dir_all(&index_dir).ok();

        // 构建 tantivy schema
        let mut schema_builder = Schema::builder();
        let file_id_field = schema_builder.add_u64_field("file_id", STORED);
        let location_field = schema_builder.add_u64_field("location", STORED);
        let text_options = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("default")
                    .set_index_option(IndexRecordOption::WithFreqsAndPositions),
            )
            .set_stored();
        let body_field = schema_builder.add_text_field("body", text_options);
        let schema = schema_builder.build();

        let index_path = index_dir.join("index");
        let meta_path = index_dir.join("meta");
        std::fs::create_dir_all(&index_path).ok();
        std::fs::create_dir_all(&meta_path).ok();

        // 打开或创建索引
        let index = Index::open_in_dir(&index_path).unwrap_or_else(|_| {
            Index::create_in_dir(&index_path, schema.clone()).expect("无法创建索引目录")
        });
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .expect("Failed to create reader");
        let writer = Arc::new(Mutex::new(index.writer(50_000_000).expect("Failed to create writer")));
        let meta_db = sled::open(&meta_path).expect("Failed to open meta db");

        // 恢复上次根目录，并确保转换为绝对路径
        let root_dir = config.last_root.as_ref()
            .and_then(|p| {
                let path = PathBuf::from(p);
                path.canonicalize().ok().or_else(|| {
                    std::env::current_dir().ok().map(|cwd| cwd.join(&path).canonicalize().unwrap_or(cwd.join(&path)))
                })
            })
            .filter(|p| p.exists());

        let total_indexed = Self::count_indexed_files(&meta_db);

        Self {
            index,
            reader,
            writer,
            schema,
            body_field,
            file_id_field,
            location_field,
            meta_db,
            config,
            config_path,
            root_dir,
            index_state: IndexState::Idle,
            pause_flag: Arc::new(AtomicBool::new(false)),
            stop_flag: Arc::new(AtomicBool::new(false)),
            index_status: "就绪".to_string(),
            log_messages: VecDeque::with_capacity(100),
            total_indexed,
            current_processing: String::new(),
            compare_result: None,
            search_query: String::new(),
            results: vec![],
            sort_by: SortBy::Relevance,
            selected_result: None,
            selected_hit: None,
            show_settings: false,
            temp_index_dir: String::new(),
            temp_pdf_reader: String::new(),
            _watcher: None,
            _watcher_handle: None,
            progress_rx: None,
            pending_open: None,
        }
    }

    /// 统计已索引文件数量（通过元数据库中 8 字节 key 的数量）
    fn count_indexed_files(meta_db: &sled::Db) -> usize {
        meta_db
            .iter()
            .filter(|item| {
                if let Ok((key, _)) = item {
                    key.len() == 8
                } else {
                    false
                }
            })
            .count()
    }

    /// 设置索引根目录（保存绝对路径到配置）
    fn set_root_directory(&mut self, path: PathBuf) {
        let abs_path = path.canonicalize().unwrap_or_else(|_| {
            std::env::current_dir().map(|cwd| cwd.join(&path).canonicalize().unwrap_or(cwd.join(&path))).unwrap_or(path)
        });
        self.root_dir = Some(abs_path.clone());
        self.index_status = format!("已选择文件夹: {}，请开始建立索引", abs_path.display());
        self.config.last_root = Some(abs_path.to_string_lossy().to_string());
        self.save_config();
    }

    /// 将当前配置写入 JSON 文件
    fn save_config(&self) {
        if let Ok(json) = serde_json::to_string_pretty(&self.config) {
            let _ = std::fs::write(&self.config_path, json);
        }
    }

    /// 启动文件系统监控，自动索引新增/修改的文件
    fn start_watcher(&mut self) {
        let root = match &self.root_dir {
            Some(d) => d.clone(),
            None => return,
        };
        let writer = self.writer.clone();
        let meta_db = self.meta_db.clone();
        let schema = self.schema.clone();
        let body_field = self.body_field;
        let file_id_field = self.file_id_field;
        let location_field = self.location_field;

        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = notify::recommended_watcher(move |res: Result<Event, _>| {
            if let Ok(event) = res {
                if let EventKind::Modify(_) | EventKind::Create(_) = event.kind {
                    for p in event.paths {
                        let _ = tx.send(p);
                    }
                }
            }
        })
        .expect("Failed to create file watcher");
        watcher
            .watch(&root, RecursiveMode::Recursive)
            .expect("Failed to watch directory");
        self._watcher = Some(watcher);

        let handle = std::thread::spawn(move || {
            for path in rx {
                if let Ok(meta) = std::fs::metadata(&path) {
                    if meta.is_file() {
                        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
                        if matches!(ext.as_str(), "pdf" | "docx" | "txt") {
                            // 强制转为绝对路径后再更新索引
                            if let Ok(abs_path) = path.canonicalize() {
                                let _ = update_single_file(
                                    &writer,
                                    &meta_db,
                                    &abs_path,
                                    &ext,
                                    &schema,
                                    body_field,
                                    file_id_field,
                                    location_field,
                                );
                            }
                        }
                    }
                }
            }
        });
        self._watcher_handle = Some(handle);
    }

    /// 开始全量/增量索引
    fn start_indexing(&mut self) {
        if self.index_state == IndexState::Indexing {
            return;
        }
        if self.root_dir.is_none() {
            self.log("请先选择文件夹".to_string());
            return;
        }
        self.index_state = IndexState::Indexing;
        self.stop_flag.store(false, Ordering::SeqCst);
        self.pause_flag.store(false, Ordering::SeqCst);
        let root = self.root_dir.as_ref().unwrap().clone();
        let writer = self.writer.clone();
        let meta_db = self.meta_db.clone();
        let schema = self.schema.clone();
        let body_field = self.body_field;
        let file_id_field = self.file_id_field;
        let location_field = self.location_field;
        let pause_flag = self.pause_flag.clone();
        let stop_flag = self.stop_flag.clone();

        let (tx, rx) = channel::<IndexMsg>();
        self.progress_rx = Some(rx);

        // 若尚未启动监控，则启动
        if self._watcher.is_none() {
            self.start_watcher();
        }

        std::thread::spawn(move || {
            let _ = full_scan_and_index(
                &writer,
                &meta_db,
                &root,
                &schema,
                body_field,
                file_id_field,
                location_field,
                Some(tx),
                pause_flag,
                stop_flag,
            );
        });
        self.log("开始建立索引...".to_string());
    }

    fn pause_indexing(&mut self) {
        self.pause_flag.store(true, Ordering::SeqCst);
        self.index_state = IndexState::Paused;
        self.log("索引已暂停".to_string());
    }

    fn resume_indexing(&mut self) {
        self.pause_flag.store(false, Ordering::SeqCst);
        self.index_state = IndexState::Indexing;
        self.log("索引已恢复".to_string());
    }

    fn stop_indexing(&mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        self.pause_flag.store(false, Ordering::SeqCst);
        self.index_state = IndexState::Idle;
        self.total_indexed = Self::count_indexed_files(&self.meta_db);
        self.log("索引已停止".to_string());
    }

    /// 对比索引与磁盘文件的差异
    fn trigger_compare(&mut self) {
        let root = match &self.root_dir {
            Some(d) => d.clone(),
            None => return,
        };
        let (new_count, mod_count, del_count) = compare_index(&self.meta_db, &root);
        self.compare_result = Some((new_count, mod_count, del_count));
        self.log(format!(
            "对比结果：新增 {}，修改 {}，删除 {}",
            new_count, mod_count, del_count
        ));
    }

    /// 执行搜索
    fn search(&mut self) {
        self.results.clear();
        self.selected_result = None;
        self.selected_hit = None;
        let searcher = self.reader.searcher();
        let query_str = self.search_query.trim();
        if query_str.is_empty() {
            return;
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let tokenized = tokenize_query(query_str);
            let query_parser = QueryParser::for_index(&self.index, vec![self.body_field]);
            let query = query_parser.parse_query(&tokenized)
                .map_err(|e| anyhow::anyhow!("查询解析错误: {}", e))?;
            let top_docs = searcher.search(&query, &TopDocs::with_limit(500))
                .map_err(|e| anyhow::anyhow!("搜索错误: {}", e))?;

            let query_words: Vec<String> = tokenized.split_whitespace()
                .map(|s| s.to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if query_words.is_empty() {
                return Ok(());
            }

            for (score, doc_address) in top_docs {
                if let Ok(doc) = searcher.doc::<TantivyDocument>(doc_address) {
                    let file_id = doc.get_first(self.file_id_field)
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let location = doc.get_first(self.location_field)
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    let meta_key = file_id.to_le_bytes();
                    if let Ok(Some(meta_bytes)) = self.meta_db.get(&meta_key) {
                        if let Ok(meta) = bincode::deserialize::<DocMeta>(&meta_bytes) {
                            let body_text = doc.get_first(self.body_field)
                                .and_then(|v| v.as_str())
                                .unwrap_or("");

                            // 统计该文档中所有查询词的出现总次数
                            let keyword_count = query_words
                                .iter()
                                .map(|w| body_text.split(' ').filter(|&t| t == w).count())
                                .sum();

                            // 提取每个查询词的命中片段（上下文 500 字符）
                            let mut hits = Vec::new();
                            for kw in &query_words {
                                let mut start = 0;
                                let mut safety = 0;
                                while start < body_text.len() && safety < 10000 {
                                    if let Some(pos) = body_text[start..].find(kw) {
                                        let abs_pos = start + pos;
                                        // 前后各取约 500 字符上下文
                                        let begin = abs_pos.saturating_sub(250);
                                        let snippet_len = kw.len() + 500;
                                        let snippet = safe_slice(body_text, begin, snippet_len).to_string();
                                        hits.push(KeywordHit {
                                            location: location as u32,
                                            snippet,
                                            keyword: kw.clone(),
                                        });
                                        start = abs_pos + kw.len();
                                        safety += 1;
                                    } else {
                                        break;
                                    }
                                }
                            }

                            // 生成结果预览摘要（取前 500 字符）
                            let main_snippet = {
                                let snippet = safe_slice(body_text, 0, 500).to_string();
                                let mut s = if body_text.len() > 500 {
                                    format!("{}...", snippet)
                                } else {
                                    snippet
                                };
                                // 高亮关键词
                                for kw in &query_words {
                                    s = s.replace(kw, &format!("【{}】", kw));
                                }
                                s
                            };

                            self.results.push(SearchResult {
                                file_id,
                                filename: meta.filename.clone(),
                                path: meta.path.clone(),   // 已经是绝对路径
                                location,
                                snippet: main_snippet,
                                score,
                                mtime: meta.mtime,
                                keyword_count,
                                hits,
                            });
                        }
                    }
                }
            }
            Ok::<_, anyhow::Error>(())
        }));

        match result {
            Ok(Ok(())) => {
                Self::sort_results(&mut self.results, &self.sort_by);
                self.index_status = format!("找到 {} 个结果", self.results.len());
            }
            Ok(Err(e)) => {
                self.index_status = format!("搜索失败: {}", e);
                self.log(format!("搜索错误: {}", e));
            }
            Err(panic_info) => {
                let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                    s.to_string()
                } else if let Some(s) = panic_info.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                self.index_status = format!("搜索过程中发生崩溃: {}", msg);
                self.log(format!("搜索崩溃: {}", msg));
            }
        }
    }

    /// 按照指定方式排序结果
    fn sort_results(results: &mut Vec<SearchResult>, sort_by: &SortBy) {
        match sort_by {
            SortBy::Relevance => results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap()),
            SortBy::FileNameAsc => results
                .sort_by(|a, b| a.filename.to_lowercase().cmp(&b.filename.to_lowercase())),
            SortBy::FileNameDesc => results
                .sort_by(|a, b| b.filename.to_lowercase().cmp(&a.filename.to_lowercase()).reverse()),
            SortBy::DateNewest => results.sort_by(|a, b| b.mtime.cmp(&a.mtime)),
            SortBy::DateOldest => results.sort_by(|a, b| a.mtime.cmp(&b.mtime)),
            SortBy::Frequency => results.sort_by(|a, b| b.keyword_count.cmp(&a.keyword_count)),
        }
    }

    fn resort_current_results(&mut self) {
        let sort_by = self.sort_by.clone();
        Self::sort_results(&mut self.results, &sort_by);
    }

    /// 打开文件并跳转到指定位置（页码/段落）
    fn open_file_with_location(&mut self, res: &SearchResult) {
        let path = res.path.clone();
        let location = res.location;
        let query = self.search_query.clone();
        self.open_with_command(&path, location, &query);
    }

    fn open_hit_with_location(&mut self, hit: &KeywordHit, file_path: &Path) {
        let location = hit.location as u64;
        let keyword = hit.keyword.clone();
        self.open_with_command(file_path, location, &keyword);
    }

    /// 通用文件打开逻辑，PDF 会优先使用用户自定义的阅读器命令，并强制使用绝对路径
    fn open_with_command(&mut self, file_path: &Path, page: u64, keyword: &str) {
        // BUG 修复：确保传递给外部程序的路径是文件的绝对路径，防止因相对路径或当前目录变化导致找不到文件
        let abs_path = if file_path.is_absolute() {
            file_path.to_path_buf()
        } else {
            // 尝试基于根目录或当前目录拼接
            if let Some(ref root) = self.root_dir {
                root.join(file_path).canonicalize().unwrap_or_else(|_| root.join(file_path))
            } else if let Ok(cwd) = std::env::current_dir() {
                cwd.join(file_path).canonicalize().unwrap_or_else(|_| cwd.join(file_path))
            } else {
                file_path.to_path_buf()
            }
        };

        let mut opened = false;
        if abs_path.extension().map_or(false, |e| e == "pdf") {
            if let Some(ref cmd_template) = self.config.pdf_reader {
                // 路径包含空格时用引号包裹
                let cmd = cmd_template
                    .replace("{file}", &format!("\"{}\"", abs_path.display()))
                    .replace("{page}", &page.to_string())
                    .replace("{keyword}", keyword);
                let parts: Vec<&str> = cmd.split_whitespace().collect();
                if !parts.is_empty() {
                    let status = std::process::Command::new(parts[0])
                        .args(&parts[1..])
                        .spawn();
                    if status.is_ok() {
                        opened = true;
                    }
                }
            }
        }
        if !opened {
            let _ = open::that(&abs_path);
        }
        self.log(format!("打开文件: {} 位置: {}", abs_path.display(), page));
    }

    /// 记录日志
    fn log(&mut self, msg: String) {
        self.log_messages.push_back(msg);
        if self.log_messages.len() > 100 {
            self.log_messages.pop_front();
        }
    }
}

// ---------- eframe App 实现 ----------
impl eframe::App for DocSearcherApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 处理索引进度消息
        let mut progress_msgs = Vec::new();
        if let Some(ref rx) = self.progress_rx {
            while let Ok(msg) = rx.try_recv() {
                progress_msgs.push(msg);
            }
        }
        for msg in progress_msgs {
            match msg {
                IndexMsg::Progress { current, total, index } => {
                    self.current_processing =
                        format!("第 {}/{} 个文件: {}", index, total, current);
                    self.total_indexed = total;
                    self.log(format!("正在索引: {}", self.current_processing));
                }
                IndexMsg::Done => {
                    self.total_indexed = Self::count_indexed_files(&self.meta_db);
                    self.log("索引完成".to_string());
                    self.progress_rx = None;
                    self.index_state = IndexState::Idle;
                }
            }
        }

        // 顶部工具栏
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("📁 选择文件夹").clicked() {
                    if let Some(path) = FileDialog::new().pick_folder() {
                        self.set_root_directory(path);
                    }
                }
                if let Some(dir) = &self.root_dir {
                    ui.label(format!("📂 {}", dir.display()));
                }
                ui.separator();
                match self.index_state {
                    IndexState::Idle => {
                        if ui.button("▶ 开始建立索引").clicked() {
                            self.start_indexing();
                        }
                    }
                    IndexState::Indexing => {
                        if ui.button("⏸ 暂停").clicked() {
                            self.pause_indexing();
                        }
                        if ui.button("⏹ 停止").clicked() {
                            self.stop_indexing();
                        }
                    }
                    IndexState::Paused => {
                        if ui.button("▶ 恢复").clicked() {
                            self.resume_indexing();
                        }
                        if ui.button("⏹ 停止").clicked() {
                            self.stop_indexing();
                        }
                    }
                }
                ui.separator();
                if ui.button("📊 对比索引").clicked() {
                    if self.root_dir.is_some() {
                        self.trigger_compare();
                    } else {
                        self.log("请先选择文件夹".to_string());
                    }
                }
                if self.compare_result.is_some() {
                    if ui.button("🔄 应用更新").clicked() {
                        self.compare_result = None;
                        self.start_indexing();
                    }
                }
                ui.separator();
                ui.label("排序：");
                let previous_sort = self.sort_by.clone();
                egui::ComboBox::from_id_source("sort_combo")
                    .selected_text(match self.sort_by {
                        SortBy::Relevance => "🔥 相关度",
                        SortBy::FileNameAsc => "📄 文件名 A→Z",
                        SortBy::FileNameDesc => "📄 文件名 Z→A",
                        SortBy::DateNewest => "🕒 最新修改",
                        SortBy::DateOldest => "🕒 最早修改",
                        SortBy::Frequency => "🔢 词频",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.sort_by, SortBy::Relevance, "🔥 相关度");
                        ui.selectable_value(&mut self.sort_by, SortBy::FileNameAsc, "📄 文件名 A→Z");
                        ui.selectable_value(&mut self.sort_by, SortBy::FileNameDesc, "📄 文件名 Z→A");
                        ui.selectable_value(&mut self.sort_by, SortBy::DateNewest, "🕒 最新修改");
                        ui.selectable_value(&mut self.sort_by, SortBy::DateOldest, "🕒 最早修改");
                        ui.selectable_value(&mut self.sort_by, SortBy::Frequency, "🔢 词频");
                    });
                if self.sort_by != previous_sort {
                    self.resort_current_results();
                }
                ui.separator();
                if ui.button("⚙️ 设置").clicked() {
                    self.show_settings = true;
                    self.temp_index_dir = self.config.index_dir.clone();
                    self.temp_pdf_reader = self.config.pdf_reader.clone().unwrap_or_default();
                }
            });
        });

        // 主区域
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                let response = ui.add(
                    egui::TextEdit::singleline(&mut self.search_query)
                        .hint_text("输入关键词搜索...")
                        .desired_width(250.0),
                );
                if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    self.search();
                }
                if ui.button("🔍 搜索").clicked() {
                    self.search();
                }
            });
            ui.label(format!(
                "已索引文件: {} | {} | {}",
                self.total_indexed, self.current_processing, self.index_status
            ));
            if let Some((new, modi, del)) = self.compare_result {
                ui.label(format!(
                    "对比结果：新增 {} 个，修改 {} 个，删除 {} 个",
                    new, modi, del
                ));
            }

            ui.columns(2, |columns| {
                let left = &mut columns[0];
                left.heading("搜索结果");
                let mut open_later: Option<SearchResult> = None;
                let mut select_idx: Option<usize> = None;

                ScrollArea::vertical().show(left, |ui| {
                    let item_height = 60.0;
                    let available_width = ui.available_width();
                    for (i, res) in self.results.iter().enumerate() {
                        let selected = Some(i) == self.selected_result;
                        let (rect, response) = ui.allocate_exact_size(
                            Vec2::new(available_width, item_height),
                            Sense::click(),
                        );
                        let mut child_ui = ui.child_ui(rect, *ui.layout());
                        let frame = egui::Frame::group(child_ui.style())
                            .fill(if selected {
                                Color32::from_rgb(230, 240, 255)
                            } else {
                                child_ui.visuals().extreme_bg_color
                            })
                            .stroke(egui::Stroke::new(1.0, Color32::GRAY));
                        frame.show(&mut child_ui, |ui| {
                            ui.set_min_height(item_height);
                            ui.vertical(|ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(format!("{}. {}", i + 1, res.filename))
                                            .strong()
                                            .color(Color32::from_rgb(0, 100, 200)),
                                    );
                                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                        ui.label(format!("🔢 {}次", res.keyword_count));
                                        if ui.button("📍").clicked() {
                                            open_later = Some(res.clone());
                                        }
                                    });
                                });
                                ui.label(format!("{} | 分数: {:.2}", res.snippet, res.score));
                            });
                        });
                        if response.clicked() {
                            select_idx = Some(i);
                        }
                    }
                });

                if let Some(res) = open_later {
                    self.open_file_with_location(&res);
                }
                if let Some(idx) = select_idx {
                    self.selected_result = Some(idx);
                    self.selected_hit = None;
                }

                let right = &mut columns[1];
                if let Some(idx) = self.selected_result {
                    if let Some(res) = self.results.get(idx) {
                        right.heading("关键词位置与预览");
                        let file_path = res.path.clone();
                        let mut open_hit: Option<KeywordHit> = None;
                        let mut select_hit: Option<usize> = None;

                        ScrollArea::vertical().id_source("right_scroll").show(right, |ui| {
                            for (hi, hit) in res.hits.iter().enumerate() {
                                let selected = Some(hi) == self.selected_hit;
                                let frame = egui::Frame::group(ui.style())
                                    .fill(if selected {
                                        Color32::from_rgb(240, 240, 220)
                                    } else {
                                        ui.visuals().extreme_bg_color
                                    })
                                    .stroke(egui::Stroke::new(1.0, Color32::GRAY));
                                let inner = frame.show(ui, |ui| {
                                    ui.set_min_width(ui.available_width() - 10.0);
                                    ui.horizontal(|ui| {
                                        ui.label(format!("{} [{}]", hit.keyword, hit.location));
                                        if ui.button("📍 打开").clicked() {
                                            open_hit = Some(hit.clone());
                                        }
                                        if ui.button("🔍").clicked() {
                                            select_hit = Some(hi);
                                        }
                                    });
                                    ui.label(&hit.snippet);
                                });
                                if inner.response.clicked() {
                                    select_hit = Some(hi);
                                }
                            }

                            let active_hit = select_hit.or(self.selected_hit);
                            if let Some(hi) = active_hit {
                                if let Some(hit) = res.hits.get(hi) {
                                    ui.separator();
                                    ui.label("预览片段 (高亮)：");
                                    let highlighted = hit.snippet.replace(
                                        &hit.keyword,
                                        &format!("【{}】", hit.keyword),
                                    );
                                    ui.label(highlighted);
                                }
                            }
                        });

                        if let Some(hit) = open_hit {
                            self.open_hit_with_location(&hit, &file_path);
                        }
                        if let Some(hi) = select_hit {
                            self.selected_hit = Some(hi);
                        }
                    }
                } else {
                    right.label("请选择一个结果以查看关键词位置");
                }
            });
        });

        // 底部日志面板
        egui::TopBottomPanel::bottom("log_panel")
            .resizable(true)
            .min_height(80.0)
            .show(ctx, |ui| {
                ui.label("操作日志：");
                ScrollArea::vertical()
                    .id_source("log_scroll")
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for msg in self.log_messages.iter().rev() {
                            ui.label(msg);
                        }
                    });
            });

        // 设置窗口
        if self.show_settings {
            egui::Window::new("设置")
                .collapsible(false)
                .show(ctx, |ui| {
                    ui.label("索引存储目录：");
                    ui.horizontal(|ui| {
                        ui.text_edit_singleline(&mut self.temp_index_dir);
                        if ui.button("浏览...").clicked() {
                            if let Some(dir) = FileDialog::new().pick_folder() {
                                self.temp_index_dir = dir.to_string_lossy().to_string();
                            }
                        }
                    });
                    ui.label("PDF 阅读器命令行模板：");
                    ui.text_edit_singleline(&mut self.temp_pdf_reader);
                    ui.label("占位符: {file} {page} {keyword} （路径自动加引号）");
                    if ui.button("保存设置").clicked() {
                        self.config.index_dir = self.temp_index_dir.clone();
                        if !self.temp_pdf_reader.is_empty() {
                            self.config.pdf_reader = Some(self.temp_pdf_reader.clone());
                        } else {
                            self.config.pdf_reader = None;
                        }
                        self.save_config();
                        self.show_settings = false;
                        self.log("设置已保存".to_string());
                    }
                    if ui.button("取消").clicked() {
                        self.show_settings = false;
                    }
                });
        }

        ctx.request_repaint_after(std::time::Duration::from_millis(200));
    }
}

// ---------- 索引辅助函数 ----------
/// 根据文件路径从元数据库中获取文件 ID
fn get_file_id(meta_db: &sled::Db, path: &Path) -> Result<Option<u64>> {
    let key = path.to_str().context("invalid path")?.as_bytes();
    if let Ok(Some(val)) = meta_db.get(key) {
        let bytes: [u8; 8] = val
            .as_ref()
            .try_into()
            .map_err(|_| anyhow::anyhow!("invalid id length"))?;
        Ok(Some(u64::from_le_bytes(bytes)))
    } else {
        Ok(None)
    }
}

/// 保存文件路径到 ID 的映射
fn set_file_id(meta_db: &sled::Db, path: &Path, file_id: u64) -> Result<()> {
    meta_db.insert(
        path.to_str().context("invalid path")?.as_bytes(),
        &file_id.to_le_bytes(),
    )?;
    Ok(())
}

/// 生成下一个可用的文件 ID
fn next_file_id(meta_db: &sled::Db) -> Result<u64> {
    let id_key = b"next_file_id";
    let id = meta_db
        .update_and_fetch(id_key, |old| {
            let old_id = old
                .map(|b| u64::from_le_bytes(b.try_into().unwrap_or([0; 8])))
                .unwrap_or(0);
            Some((old_id + 1).to_le_bytes().to_vec())
        })?
        .map(|b| u64::from_le_bytes(b.as_ref().try_into().unwrap_or([0; 8])))
        .unwrap_or(1);
    Ok(id)
}

/// 更新单个文件的索引（接收绝对路径）
fn update_single_file(
    writer: &Arc<Mutex<IndexWriter>>,
    meta_db: &sled::Db,
    path: &Path,
    ext: &str,
    _schema: &Schema,
    body_field: Field,
    file_id_field: Field,
    location_field: Field,
) -> Result<()> {
    // 再次确保使用绝对路径
    let abs_path = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let mut writer = writer.lock().unwrap_or_else(|e| e.into_inner());
    // 若已索引，先删除旧的索引文档
    if let Ok(Some(file_id)) = get_file_id(meta_db, &abs_path) {
        let term = tantivy::Term::from_field_u64(file_id_field, file_id);
        writer.delete_term(term);
        meta_db.remove(&file_id.to_le_bytes())?;
    }
    let pages = extract_pages_safe(&abs_path, ext)?;
    let mtime = std::fs::metadata(&abs_path)?
        .modified()?
        .duration_since(UNIX_EPOCH)?
        .as_secs();
    let md5 = compute_md5(&abs_path)?;
    let filename = abs_path.file_name().unwrap().to_str().unwrap().to_string();
    let file_id = if let Ok(Some(id)) = get_file_id(meta_db, &abs_path) {
        id
    } else {
        next_file_id(meta_db)?
    };
    set_file_id(meta_db, &abs_path, file_id)?;
    let meta = DocMeta {
        path: abs_path,
        filename,
        mtime,
        md5,
    };
    meta_db.insert(&file_id.to_le_bytes(), bincode::serialize(&meta)?)?;
    for (loc, text) in pages {
        if text.trim().is_empty() {
            continue;
        }
        let tokenized_text = tokenize(&text);
        let mut doc = TantivyDocument::default();
        doc.add_u64(file_id_field, file_id);
        doc.add_u64(location_field, loc as u64);
        doc.add_text(body_field, tokenized_text);
        writer.add_document(doc)?;
    }
    writer.commit()?;
    Ok(())
}

/// 全量/增量遍历根目录并索引，支持暂停和停止
fn full_scan_and_index(
    writer: &Arc<Mutex<IndexWriter>>,
    meta_db: &sled::Db,
    root: &Path,
    schema: &Schema,
    body_field: Field,
    file_id_field: Field,
    location_field: Field,
    progress_tx: Option<Sender<IndexMsg>>,
    pause_flag: Arc<AtomicBool>,
    stop_flag: Arc<AtomicBool>,
) -> Result<()> {
    // 统计总文件数
    let mut total_files = 0;
    for entry in WalkDir::new(root).follow_links(true).into_iter().filter_map(|e| e.ok()) {
        if entry.file_type().is_file() {
            let ext = entry.path().extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
            if matches!(ext.as_str(), "pdf" | "docx" | "txt") {
                total_files += 1;
            }
        }
    }

    let mut file_index = 0;
    for entry in WalkDir::new(root).follow_links(true).into_iter().filter_map(|e| e.ok()) {
        // 检查停止标志
        if stop_flag.load(Ordering::SeqCst) {
            break;
        }
        while pause_flag.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(100));
            if stop_flag.load(Ordering::SeqCst) {
                break;
            }
        }

        if !entry.file_type().is_file() {
            continue;
        }
        let ext = entry.path().extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
        if !matches!(ext.as_str(), "pdf" | "docx" | "txt") {
            continue;
        }

        // entry.path() 已是绝对路径（因为 root 是绝对路径）
        let abs_path = entry.path().to_path_buf();
        file_index += 1;
        if let Some(ref tx) = progress_tx {
            let _ = tx.send(IndexMsg::Progress {
                current: abs_path.display().to_string(),
                total: total_files,
                index: file_index,
            });
        }

        // 快速检查元数据，跳过未修改的文件
        let current_mtime = std::fs::metadata(&abs_path)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let current_md5 = compute_md5(&abs_path).unwrap_or_default();

        if let Ok(Some(file_id)) = get_file_id(meta_db, &abs_path) {
            if let Ok(Some(meta_bytes)) = meta_db.get(&file_id.to_le_bytes()) {
                if let Ok(old_meta) = bincode::deserialize::<DocMeta>(&meta_bytes) {
                    if old_meta.mtime == current_mtime && old_meta.md5 == current_md5 {
                        continue; // 文件未变化，跳过
                    }
                }
            }
        }

        // 更新索引
        if let Err(e) = update_single_file(
            writer, meta_db, &abs_path, &ext, schema, body_field, file_id_field, location_field,
        ) {
            // 记录错误到日志文件
            let log_dir = dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".doc_searcher");
            let _ = std::fs::create_dir_all(&log_dir);
            let error_log = log_dir.join("error.log");
            if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&error_log) {
                let _ = writeln!(f, "Index error {}: {}", abs_path.display(), e);
            }
        }
    }

    // 清理已删除的文件
    if !stop_flag.load(Ordering::SeqCst) {
        let mut writer = writer.lock().unwrap_or_else(|e| e.into_inner());
        let mut to_remove = Vec::new();
        for item in meta_db.iter() {
            if let Ok((key, value)) = item {
                if key.len() == 8 {
                    let bytes: [u8; 8] = match key.as_ref().try_into() {
                        Ok(b) => b,
                        Err(_) => continue,
                    };
                    let file_id = u64::from_le_bytes(bytes);
                    if let Ok(meta) = bincode::deserialize::<DocMeta>(&value) {
                        if !meta.path.exists() {
                            to_remove.push((file_id, meta.path.clone()));
                        }
                    }
                }
            }
        }
        for (file_id, path) in to_remove {
            let term = tantivy::Term::from_field_u64(file_id_field, file_id);
            writer.delete_term(term);
            meta_db.remove(&file_id.to_le_bytes())?;
            if let Some(p) = path.to_str() {
                meta_db.remove(p.as_bytes())?;
            }
        }
        writer.commit()?;
    }

    if let Some(tx) = progress_tx {
        let _ = tx.send(IndexMsg::Done);
    }
    Ok(())
}

/// 对比索引元数据与磁盘文件，返回 (新增, 修改, 删除)
fn compare_index(meta_db: &sled::Db, root: &Path) -> (usize, usize, usize) {
    let mut new_count = 0;
    let mut mod_count = 0;
    let mut del_count = 0;

    let mut disk_files: HashMap<PathBuf, (u64, String)> = HashMap::new();
    for entry in WalkDir::new(root).follow_links(true).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path().to_path_buf();
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
        if !matches!(ext.as_str(), "pdf" | "docx" | "txt") {
            continue;
        }
        let mtime = std::fs::metadata(&path)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let md5 = compute_md5(&path).unwrap_or_default();
        disk_files.insert(path, (mtime, md5));
    }

    let mut db_files: HashMap<PathBuf, (u64, String)> = HashMap::new();
    for item in meta_db.iter() {
        if let Ok((key, value)) = item {
            if key.len() == 8 {
                if let Ok(meta) = bincode::deserialize::<DocMeta>(&value) {
                    db_files.insert(meta.path.clone(), (meta.mtime, meta.md5.clone()));
                }
            }
        }
    }

    for (path, (mtime, md5)) in &disk_files {
        if let Some((db_mtime, db_md5)) = db_files.get(path) {
            if db_mtime != mtime || db_md5 != md5 {
                mod_count += 1;
            }
        } else {
            new_count += 1;
        }
    }
    for path in db_files.keys() {
        if !disk_files.contains_key(path) {
            del_count += 1;
        }
    }
    (new_count, mod_count, del_count)
}

// ---------- 程序入口 ----------
fn main() -> Result<()> {
    // 全局 panic 钩子，记录崩溃信息到 error.log
    let log_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".doc_searcher");
    std::fs::create_dir_all(&log_dir).ok();
    let error_log = log_dir.join("error.log");
    std::panic::set_hook(Box::new(move |info| {
        let msg = format!(
            "程序崩溃: {}\n回溯: {:?}\n",
            info,
            std::backtrace::Backtrace::capture()
        );
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&error_log) {
            let _ = f.write_all(msg.as_bytes());
        }
        eprintln!("{}", msg);
    }));

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1100.0, 750.0]),
        ..Default::default()
    };

    eframe::run_native(
        "本地文档检索系统",
        options,
        Box::new(|cc| {
            // 加载中文字体（Windows 下尝试微软雅黑、宋体）
            let mut fonts = egui::FontDefinitions::default();
            if cfg!(target_os = "windows") {
                let font_paths = [
                    "C:\\Windows\\Fonts\\msyh.ttc",
                    "C:\\Windows\\Fonts\\msyhbd.ttc",
                    "C:\\Windows\\Fonts\\simsun.ttc",
                ];
                for path in &font_paths {
                    if let Ok(bytes) = std::fs::read(path) {
                        fonts.font_data.insert(
                            "chinese".to_owned(),
                            egui::FontData::from_owned(bytes.into()),
                        );
                        fonts
                            .families
                            .entry(egui::FontFamily::Proportional)
                            .or_default()
                            .insert(0, "chinese".to_owned());
                        fonts
                            .families
                            .entry(egui::FontFamily::Monospace)
                            .or_default()
                            .insert(0, "chinese".to_owned());
                        break;
                    }
                }
            }
            cc.egui_ctx.set_fonts(fonts);
            Box::new(DocSearcherApp::new(cc))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {}", e))?;

    Ok(())
}