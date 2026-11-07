# Выбор публичной модели отделения вокала

Проверено: **7 ноября 2026 года**. Область: CPU-only FP32, `audio-separator==0.48.0`, отделение **всего вокала от инструментала**. Только опубликованные оценки и первичные источники; веса и датасеты не скачивались, inference и локальные бенчмарки не запускались. Изменён только этот новый исследовательский файл.

## Краткий вывод

**Рекомендация по качеству из проверенных публично скачиваемых моделей — unwa / pcunwa BS-Roformer Leap Xe, именно `Xe/bs_leap_xe_voc.ckpt`.** Основание: MVSep Multisong **SDR vocals 11.7920 / instrumental 18.0995**, один checkpoint, FP32, без TTA, overlap 2; instrumental получен вычитанием вокала из смеси. Это не ансамбль и не закрытая модель сервиса. Есть дополнительная оценка тех же весов через зеркало: 11.7615 / 18.0689. [1][2][3]

Это **не безусловное предложение заменить default прямо сейчас**:

1. В авторском репозитории Leap **не указана лицензия весов**; вопрос `LICENSE missing` остаётся без ответа. Публичная загрузка не означает разрешения на встраивание, распространение или коммерческое применение. [4][5]
2. Семейство и параметры соответствуют реализации BS-Roformer в 0.48.0, но загрузка конкретного checkpoint здесь не проверялась. Нужна существующая в приложении интеграция локального checkpoint/config: в проверенном статическом `models.json` библиотеки Leap не зарегистрирован. [6][7][8]
3. Опубликованные цифры относятся примерно к **20 секундам контекста**, а не к зафиксированным в приложении 801 кадру. Есть дополнительное несовпадение между `audio.chunk_size` и `inference.dim_t` в авторском YAML; простое отключение override не воспроизводит оценённый режим. Подробности ниже. [1][6][9]

Если обязательны **нынешний код без изменений и явно заявленная лицензия**, разумно **сохранить текущий Kimberley Jensen**. Это практический fallback, а не утверждение, что он качественнее Leap. У закреплённой ревизии Kimberley заявлена MIT; опубликованные оценки по-прежнему конкурентны, но уже не лучшие среди исследованных новых моделей. [10][11][12]

## Как интерпретировались оценки

- Главный общий ориентир — **один и тот же MVSep Multisong**, а не SDR из имени файла. Это 100 минутных фрагментов разных жанров, 44.1 kHz; эталонные stems скрыты. MVSep сам предупреждает о возможном пересечении с обучающими данными отдельных моделей. [13]
- MVSep **Synth** — другой набор: случайно смешанные вокальные и инструментальные фрагменты, также 100 × 1 минута. Его цифры нельзя напрямую сравнивать с Multisong, MUSDB18HQ, MDX23 или произвольной авторской validation. [13]
- MVSep проверяет **загруженные результаты разделения**. Название алгоритма, описание настроек и отметка «ensemble» подаются участником. Численный результат первичен для оценочного сервера, но описание не является независимой экспертизой кода или аудиторским подтверждением воспроизводимости. [13]
- Таблицы обновляются. В текущем Multisong сверху есть `Original stems` в WAV/MP3/M4A: это контроль качества кодеков, **не модели разделения**. Их нельзя включать в рейтинг нейросетей. [14]
- Даже в рамках одного датасета различаются длина контекста, overlap, обработка границ, нормализация, вывод инструментала и формат экспорта. Таблица ниже сравнивает **опубликованные системы/настройки**, а не изолированное влияние весов при одинаковом inference.

## Сравнение на Multisong

Все SDR в этой таблице — **только MVSep Multisong**; выше лучше. Округление сохранено из первичных источников.

| Модель / опубликованный вариант | Vocals SDR | Instrumental SDR | Что это доказывает / ограничивает |
| --- | ---: | ---: | --- |
| **Leap Xe vocals, FP32, no TTA, overlap 2**, запись 10846 | **11.7920** | **18.0995** | Основное основание рекомендации; `chunk_size=881559`, batch 2, instrumental = mixture − vocals. [1] |
| Те же Leap Xe vocals через переименованное зеркало, 10411 | 11.7615 | 18.0689 | LFS SHA-256 совпадает с авторскими весами; настройки полностью не раскрыты. [2][3] |
| Leap Xe, запись 10178, подтверждённая автором | 11.7577 | 17.5303 | Автор pcunwa подтвердил соответствие Xe этой записи. Не подменять её instrumental цифрой из residual-прогона 10846. [15][16] |
| becruily **Deux**, 9482 | 11.3695 | 17.5466 | Один **двухвыходный** Mel-Band RoFormer, не ансамбль. Описание результата — `Half split`; те же числа приведены автором в карточке. CC-BY-NC-4.0. [17][18][19] |
| unwa MelBand **Big Beta 7**, 10516 | 11.1179 | 17.4253 | Публичный результат через зеркало; идентичность авторских весов проверена по LFS SHA-256. Небольшой отрыв от Kimberley, существенно ниже Leap Xe. [20][21] |
| unwa MelBand **Big Beta 6X**, 10582 | 11.1148 | 17.4223 | Не выбирать только потому, что «Big» или файл больше. [21][22] |
| Kimberley, overlap 2, `dim_t=1101`, 6672 | 11.0742 | 17.3817 | Более длинный контекст, чем нынешний app default. [23] |
| Kimberley, overlap 2, `dim_t=801`, 6673 | 11.0021 | 17.3096 | Наиболее близкие явно описанные параметры к текущему приложению; **не подтверждение идентичности всего pipeline**. [11] |
| Kimberley в curated MSST pretrained table | 10.98 | — | Авторитетная общая таблица прямо указывает Multisong, но не раскрывает все параметры этой строки. [12] |
| ViperX BS ep317 `sdr_12.9755` в MSST | 10.87 | — | **12.9755 в имени — не Multisong SDR**. Hosted-оценка viperx 6176: 10.8717 / 17.1791. [12][24] |
| Старый becruily vocals, обычные публичные результаты | ~10.50–10.55 | ~16.82–16.86 | Не путать с Deux. Встречается 10.8222 `Later epoch`, но без доказанной привязки к текущему доступному файлу — не переносим этот результат на него. [25][26] |

Наблюдаемый разрыв Leap Xe 10846 относительно Kimberley 6673 составляет около **+0.79 dB vocals SDR**. Это разница опубликованных результатов на одном наборе, **не обещанный прирост текущего приложения**: условия и реализации различаются. Аналогично сравнение с MSST 10.98 — не чистый controlled experiment. [1][11][12]

### Почему не взять абсолютную верхушку leaderboard

На текущем Multisong **BS Roformer 124 bands (MVSep, 2026.07.10)** имеет 12.3339 / 18.6414; BS PolarFormer 124 bands — 12.0230 / 18.3304; есть также ансамбли. Но для этих конкретных лидирующих весов в проверенных первичных страницах **не найден подтверждённый публичный checkpoint с лицензией**, подходящий для локальной установки. Они рассматриваются как hosted-варианты, а не как доступные пользователю веса. Это не утверждение, что любой checkpoint MVSep закрыт. [14][27][28]

Отдельная ловушка: MSST действительно публикует **BS PolarFormer float16**, но его строка даёт **11.00** vocals SDR на Multisong. Нет основания считать эти веса тем же checkpoint, что hosted PolarFormer с 11.7575 или 12.0230. В 0.48.0 проверенный RoformerLoader различает BS-Roformer и Mel-Band-RoFormer, но не PolarFormer; это не очевидная drop-in замена. [7][12][14]

Старый Synth лидер BS Roformer MVSep 2025.07 имеет **14.5837 / 14.2875**, тогда как тот же hosted-вариант на Multisong — 11.8938 / 18.2013. Это наглядная причина не сравнивать число «14.58» с «11.79» Leap без названия датасета. [14][29]

## Качество вокала и качество минусовки — разные цели

Цель данной рекомендации — универсальное разделение **all vocals / instrumental**, не выделение только lead vocals, не dereverb и не `other` из четырёх stems.

- **Fullness и bleedless конфликтуют:** сохранение тихих гармоник/дыхания/хвостов может оставить больше постороннего звука. Сам MVSep помечает старые becruily vocals/instrumental, Big Beta 5e и некоторые другие версии как high/max fullness, **but noisy**. Более «полный» вокал не обязательно чище. [28]
- В записи Leap Xe 10846: vocals bleedless **38.0739**, fullness **18.1440**; в Deux 9482: **28.9679 / 23.2490**. Для инструментала: Leap **45.8545 / 29.5615**, Deux **41.3606 / 34.2534**. Эти оценки иллюстрируют компромисс чистота/полнота, а не заменяют прослушивание; нельзя выводить «Deux хуже на каждом треке». [1][17]
- Автор Leap публикует **разные** `bs_leap_xe_voc.ckpt` и `bs_leap_xe_inst.ckpt`: в первом `target_instrument: vocals`, во втором `other`. Рекомендован **voc** + остаток смеси, потому что именно этот вариант имеет наиболее ясно описанную сильную FP32/no-TTA оценку и соответствует нынешней задаче приложения. [1][6][30]
- В residual-схеме ошибки вокала и инструментала связаны: если истинная смесь аддитивна и instrumental = mixture − estimated vocals, ошибка второго stem — отрицание ошибки первого. Поэтому высокие SDR обоих residual-stems не являются двумя независимыми доказательствами. Двухвыходная модель или отдельный instrumental checkpoint может иметь иной баланс.
- Lead/back karaoke-модели оцениваются на другой задаче: они могут намеренно оставлять бэк-вокал в «минусовке». Их lead-vocal SDR нельзя переносить на задачу удаления всего вокала. MVSep документирует karaoke отдельно. [28]
- HTDemucs `other` — остаточная инструментальная группа без vocals/bass/drums, а не «вся мелодия» и не instrumental. У MSST HTDemucs FT Other 5.85 относится именно к `other`; эти веса и назначение не заменяются выбором вокального checkpoint. [12][13]

## Проверка происхождения checkpoint

### Рекомендуемая пара Leap Xe

Авторский репозиторий: [pcunwa/BS-Roformer-Leap](https://huggingface.co/pcunwa/BS-Roformer-Leap).

Зафиксированная для этой заметки ревизия: **`4e47d6662ae82eaa8b4ac4329fe66099a843b48e`**. Это не совет автоматически следовать `main`. Метаданные получены через HF API; сами веса не загружались. [3]

| Поле | Значение |
| --- | --- |
| Веса | `Xe/bs_leap_xe_voc.ckpt` |
| Размер файла | 267,796,851 байт — размер checkpoint, **не peak RAM** |
| LFS SHA-256 | `b739c1d2d87a81cd3dd3844ed9ad0bd678708c7a0a761a03a1aaff9af79a096d` |
| Config | `Xe/leap_xe_config_voc.yaml` |
| Git blob config | `dacf4b01f0639d17cf61be1624e41be786258508` |
| Параметры | stereo 44.1 kHz, dim 256, depth 16, time/freq depth 1/1, STFT FFT 2048, **model hop 512**, один target `vocals` |

Автор **лично подтвердил**: стандартный Leap — MVSep 10146, **Xe — 10178**; прогоны выполнил anvuew. В той же дискуссии anvuew пишет, что менял overlap на 4, но сама запись 10178 подписана `overlap 2`. Это противоречие оставлено открытым: нельзя выдавать ранний прогон за полностью воспроизводимую спецификацию. Более поздний 10846 явно документирует overlap 2, FP32 и отсутствие TTA, но это описание другого участника, не автора весов. [1][15][16]

Запись 10411 ссылается на зеркало `noblebarkrr/mvsepless_resources/bs_roformer/bs_leap_xe_voc_unwa.ckpt`. HF LFS metadata зеркала совпадает с авторским **SHA-256 и размером весов**, а Git blob config — с авторским YAML. Это позволяет связать дополнительную оценку с теми же весами, несмотря на переименование. Метаданные — не результат локального хеширования скачанного файла. [2][3][31]

Для Big Beta 7 также проверено совпадение авторского `big_beta7.ckpt` и зеркального `mbr_bigbeta7_unwa.ckpt`: SHA-256 **`9d68b9a8689a3500c45d3418a7811934e557760db07285f483088a0965b0eb88`**, размер 944,675,923 байт. Для Beta 6X: **`e16d702f4e20f13d60b293541c1dea75cb4414a5846b36780e28ef70352a4e5c`**. Но зеркальные YAML имеют другие blob ID: совпадение весов **не подтверждает одинаковые настройки inference**. [21][32]

Для Deux авторский и зеркальный checkpoint тоже совпадают: SHA-256 **`10255c02295bf3e3865d4ee50ff752d7b19b124ed5fd93b147babc4333eda3aa`**; авторская ревизия **`402dd9a19baaed8c6f1b54458ce6ae2ae19f354e`**. Слово «Dual» здесь означает `num_stems: 2`, а не два запуска двух разных нейросетей. [19][32][43]

Текущий Kimberley pinned revision действительно остаётся **`ac9b0614ab3cd7f77219e18ba494dfd93956c348`**, файл `MelBandRoformer.ckpt`, LFS SHA-256 **`87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e`**. MSST связывает свою строку Kimberley именно с этим авторским HF-репозиторием. Оценки старых записей MVSep не содержат hash, поэтому их привязка к файлу слабее, чем проверенная связь авторских и зеркальных Leap-весов. [10][12]

## Лицензирование весов

Лицензия `audio-separator`, MSST или кода архитектуры **не подменяет лицензию каждого checkpoint**.

| Веса | Что удалось подтвердить |
| --- | --- |
| **Leap / Leap Xe** | Нет `license` в HF metadata, нет LICENSE в дереве, карточка содержит только изображение. Открытый вопрос «LICENSE missing» от 04.11.2026 без ответа. Не маркировать MIT/Apache/CC и не считать коммерчески разрешённым. Нужны условия правообладателя. [3][4][5] |
| **unwa MelBand Big beta** | В проверенном авторском дереве и metadata нет лицензии/карточки с условиями. Лицензия исходного Kimberley не доказывает права на изменённые веса и добавленные обучающие данные. [21] |
| **becruily Deux** | Авторская карточка явно указывает **CC-BY-NC-4.0**: attribution и noncommercial; это не универсально разрешённый default для коммерческого продукта. [18] |
| Старые becruily vocals / instrumental | В проверенных авторских metadata/деревьях нет заявленной лицензии; не переносить CC-BY-NC Deux на другие репозитории без заявления автора. [33][34] |
| **Kimberley pinned** | Карточка на точной закреплённой ревизии содержит **`license: mit`**. В старой дискуссии автор сначала давал разрешение Audacity, затем объявлял GPL-3.0; текущая MIT-метка новее. При распространении сохранять доказательство лицензии именно закреплённой ревизии; полноценного LICENSE-файла в HF-дереве нет. Это подтверждение заявленной лицензии, не аудит прав на training data. [10][35][36] |
| ViperX ep317 | Источник файла подтверждён через curated MSST → TRvlvr public release, но отдельные первичные условия лицензии именно checkpoint в просмотренных источниках не установлены. Имеющийся в приложении restricted-license checkbox — не первоисточник лицензии. Не считать MIT автоматически по лицензии wrapper/репозитория. [12] |

Неясная лицензия Leap — **реальное ограничение рекомендации**, а не деталь, которую можно закрыть текущим флажком ViperX. Наличие доступного файла само по себе не даёт уверенного правового основания для включения весов в поставку.

## Совместимость именно с audio-separator 0.48.0 и CPU

Статическая проверка выполнена по **тегу `v0.48.0`**, не по текущему `main`.

- В поставке есть реализации **BSRoformer** и **MelBandRoformer**; loader создаёт модель по config, читает state dict через `torch.load(..., weights_only=True)`, затем `to(device).eval()`. Leap Xe использует параметры, присутствующие в BS-конструкторе: dim/depth, список bands, time/freq depths, hop, `mlp_expansion_factor`, `skip_connection: false`. Явного требования CUDA/custom GPU kernel в этом config нет. Это **совместимость по исходникам**, не проверенный запуск конкретного файла. [6][7][37]
- `flash_attn: true` здесь включает PyTorch SDPA; реализация имеет CPU backend, включая math path. Это **не обязательная установка CUDA FlashAttention**. Поле `training.use_amp: true` в YAML само по себе не переключает precision у `audio-separator`: у wrapper отдельный FP32 путь и runtime-настройки autocast/native FP16. Важно: MSST, несмотря на расположение поля в `training`, действительно читает его и во время inference; нельзя считать любой запуск авторского YAML автоматически FP32. Запись 10846 явно заявляет FP32. [1][6][38][39][44]
- Новый checkpoint не включён в проверенный статический `models.json`, а штатный `Separator.load_model()` ищет имя в каталоге. Поэтому наличие подходящей архитектуры не означает, что `load_model("bs_leap_xe_voc.ckpt")` автоматически заработает без регистрации/существующей project-specific загрузки. Эта заметка **не предлагает строить updater или менять инфраструктуру**. [8][39]
- Нет проверенных здесь первичных измерений **CPU FP32 latency / peak RAM** Leap Xe на целевой машине. По размеру checkpoint нельзя обещать ни определённое время, ни меньший расход RAM; длинный контекст увеличивает нагрузку attention. Число 801 и batch 1 — также не доказательство пригодности большого контекста по памяти.

### Критичное различие длины сегмента

Из переданной локальной инспекции: `VocalEngine` использует `segment_size=801`, `override_model_segment_size=True`, batch 1, default overlap 2; `worker.py` сообщает фиксированные 801 кадр / 8 секунд. Это контекст задачи, не повторно проведённая локальная проверка.

В 0.48.0 MDXC Roformer выводит длину чанка **не из `audio.chunk_size`**, а так:

`chunk_samples = model.stft_hop_length × (segment_frames − 1)`

При override=False берётся `inference.dim_t`; при True — переданный `segment_size`. Batch size в этой ветке, по комментарию и циклу реализации, фактически не используется для пакетирования Roformer-окон. Поэтому несовпадение batch 1/2 само по себе менее важно, чем длина/overlap/алгоритм перекрытия. [9]

| Режим | Фактический контекст по формуле 0.48.0 |
| --- | --- |
| Текущий Kimberley, 801, hop 441 | `441 × 800 = 352800`, ровно **8 s**; совпадает с авторским training chunk. [9][40] |
| Leap Xe с нынешними 801, **model hop 512** | `512 × 800 = 409600`, примерно **9.288 s**, **не 8 s** |
| Leap Xe, override=False, авторское `inference.dim_t=1876` | `512 × 1875 = 960000`, примерно **21.769 s** |
| Опубликованный FP32/no-TTA Leap Xe 10846 | `audio.chunk_size=881559`, примерно **19.990 s** [1][6] |

В Leap YAML `audio.hop_length=441` помечен как нерабочее поле; реальный STFT hop в `model` — 512. Нельзя менять его на 441 только ради совпадения с приложением: это параметр архитектуры/предобработки весов, а не просто настройка скорости. **881559 не кратно 512**, поэтому нынешняя формула целочисленных frames сама по себе не задаёт в точности тот же размер чанка. [6][9]

Следствие: будущий переход потребует **осознанно согласовать config-driven context и backend semantics**, а не только заменить URL весов или выключить override. Даже после этого исходный MVSep результат не гарантирован: 0.48.0 имеет собственные Hamming overlap-add и расписание окон. Для задачи «выбрать по опубликованным анализам» этого ограничения достаточно — строить локальный benchmark не требуется. [9]

Для Big Beta 7 авторское `audio.chunk_size=676935`, `inference.dim_t=1201`, hop 441: native MSST и frame-based wrapper тоже задают разные длины. MSST берёт `inference.chunk_size`, если он задан, иначе `audio.chunk_size`, а не wrapper-поле `inference.dim_t`. Для Deux компактный `config.yaml` вообще не содержит нужного wrapper полю `inference.dim_t`; рядом опубликован более полный `config_deux_becruily.yaml` с 1101, но это также не тождественно `audio.chunk_size=573300`. Нельзя без проверки использовать «любой yaml рядом с ckpt». [19][41][42][44]

Нынешняя project-specific настройка `target_instrument: vocals` концептуально подходит Leap Xe voc. **Не подходит без адаптации Deux**, у которого `target_instrument: null`, два стема `Vocals` / `Instrumental` и соответствующий `num_stems: 2`. Принудительная подмена target — не корректная настройка двухвыходных весов. [6][19]

GUI default по insertion order и checkbox, привязанный исключительно к ViperX, — ограничения из переданной инспекции проекта. Ни их менять, ни создавать auto-update инфраструктуру исследование не разрешает. Здесь не менялись engine, worker, GUI, модели или существующая документация.

## Итоговое решение

**Для следующей модели по качеству выбрать Leap Xe vocals**: лучшее сочетание сильной опубликованной оценки, подтверждения автором происхождения семейства checkpoint, проверяемой идентичности зеркальных весов и привычной поддерживаемой архитектуры среди рассмотренных кандидатов. **Не объявлять «лучшей вообще» и не обещать 11.792 SDR при нынешних 801 кадре.**

**Для немедленного сохранения рабочего и лицензированного default — оставить закреплённый Kimberley**, пока не уточнена лицензия Leap и не согласована семантика сегментов. Не переходить на ViperX только из-за `12.9755` в имени; не выбирать Big/beta/Deux только по новизне, популярности или размеру файла.

## Первичные источники

[1]: https://mvsep.com/quality_checker/entry/10846
[2]: https://mvsep.com/quality_checker/entry/10411
[3]: https://huggingface.co/api/models/pcunwa/BS-Roformer-Leap?blobs=true
[4]: https://huggingface.co/pcunwa/BS-Roformer-Leap
[5]: https://huggingface.co/pcunwa/BS-Roformer-Leap/discussions/4
[6]: https://huggingface.co/pcunwa/BS-Roformer-Leap/raw/4e47d6662ae82eaa8b4ac4329fe66099a843b48e/Xe/leap_xe_config_voc.yaml
[7]: https://github.com/nomadkaraoke/python-audio-separator/blob/v0.48.0/audio_separator/separator/roformer/roformer_loader.py
[8]: https://github.com/nomadkaraoke/python-audio-separator/blob/v0.48.0/audio_separator/models.json
[9]: https://github.com/nomadkaraoke/python-audio-separator/blob/v0.48.0/audio_separator/separator/architectures/mdxc_separator.py#L489-L567
[10]: https://huggingface.co/api/models/KimberleyJSN/melbandroformer?blobs=true
[11]: https://mvsep.com/quality_checker/entry/6673
[12]: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md
[13]: https://mvsep.com/quality_checker
[14]: https://mvsep.com/quality_checker/multisong_leaderboard?sort=vocals
[15]: https://huggingface.co/pcunwa/BS-Roformer-Leap/discussions/2
[16]: https://mvsep.com/quality_checker/entry/10178
[17]: https://mvsep.com/quality_checker/entry/9482
[18]: https://huggingface.co/becruily/mel-band-roformer-deux/raw/402dd9a19baaed8c6f1b54458ce6ae2ae19f354e/README.md
[19]: https://huggingface.co/becruily/mel-band-roformer-deux/raw/402dd9a19baaed8c6f1b54458ce6ae2ae19f354e/config_deux_becruily.yaml
[20]: https://mvsep.com/quality_checker/entry/10516
[21]: https://huggingface.co/api/models/pcunwa/Mel-Band-Roformer-big?blobs=true
[22]: https://mvsep.com/quality_checker/multisong_leaderboard?sort=vocals&page=7
[23]: https://mvsep.com/quality_checker/entry/6672
[24]: https://mvsep.com/quality_checker/entry/6176
[25]: https://mvsep.com/quality_checker/multisong_leaderboard?sort=vocals&algo_name_filter=becruily
[26]: https://mvsep.com/quality_checker/entry/7701
[27]: https://mvsep.com/quality_checker/entry/10197
[28]: https://mvsep.com/algorithms
[29]: https://mvsep.com/quality_checker/synth_leaderboard
[30]: https://huggingface.co/pcunwa/BS-Roformer-Leap/raw/4e47d6662ae82eaa8b4ac4329fe66099a843b48e/Xe/leap_xe_config_inst.yaml
[31]: https://huggingface.co/api/models/noblebarkrr/mvsepless_resources/tree/main/bs_roformer?limit=1000
[32]: https://huggingface.co/api/models/noblebarkrr/mvsepless_resources/tree/main/mel_band_roformer?limit=1000
[33]: https://huggingface.co/api/models/becruily/mel-band-roformer-vocals?blobs=true
[34]: https://huggingface.co/api/models/becruily/mel-band-roformer-instrumental?blobs=true
[35]: https://huggingface.co/KimberleyJSN/melbandroformer/raw/ac9b0614ab3cd7f77219e18ba494dfd93956c348/README.md
[36]: https://huggingface.co/KimberleyJSN/melbandroformer/discussions/2
[37]: https://github.com/nomadkaraoke/python-audio-separator/blob/v0.48.0/audio_separator/separator/uvr_lib_v5/roformer/bs_roformer.py
[38]: https://github.com/nomadkaraoke/python-audio-separator/blob/v0.48.0/audio_separator/separator/uvr_lib_v5/roformer/attend.py
[39]: https://github.com/nomadkaraoke/python-audio-separator/blob/v0.48.0/audio_separator/separator/separator.py
[40]: https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model/blob/main/README.md
[41]: https://huggingface.co/pcunwa/Mel-Band-Roformer-big/raw/1508d1ed7c54cb0017b2cbfaabdaf3ca87d2cf74/big_beta7.yaml
[42]: https://huggingface.co/becruily/mel-band-roformer-deux/raw/402dd9a19baaed8c6f1b54458ce6ae2ae19f354e/config.yaml
[43]: https://huggingface.co/api/models/becruily/mel-band-roformer-deux?blobs=true
[44]: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/utils/model_utils.py#L124-L260
