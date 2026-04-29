/**
 * PuzzleGame - 拼图游戏核心逻辑
 * 状态编码: grid[row][col] = pieceIndex (与Python端numpy grid一致)
 * 玩法: 拖动交换两个拼图块，相邻且位置正确的块自动融合为整体
 */

// === Union-Find ===
class UnionFind {
    constructor(n) {
        this.parent = Array.from({ length: n }, (_, i) => i);
        this.rank = new Array(n).fill(0);
    }
    find(x) {
        if (this.parent[x] !== x) this.parent[x] = this.find(this.parent[x]);
        return this.parent[x];
    }
    union(a, b) {
        const ra = this.find(a), rb = this.find(b);
        if (ra === rb) return;
        if (this.rank[ra] < this.rank[rb]) { this.parent[ra] = rb; }
        else if (this.rank[ra] > this.rank[rb]) { this.parent[rb] = ra; }
        else { this.parent[rb] = ra; this.rank[ra]++; }
    }
    reset(n) {
        this.parent = Array.from({ length: n }, (_, i) => i);
        this.rank = new Array(n).fill(0);
    }
}

// === PuzzleGame ===
class PuzzleGame {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.rows = options.rows || 6;
        this.cols = options.cols || 6;
        this.onStateChange = options.onStateChange || (() => {});

        this.grid = [];          // grid[row][col] = pieceIndex
        this.pieces = [];        // pieces[i] = { image: ImageData, originalRow, originalCol }
        this.moveCount = 0;
        this.history = [];       // undo stack: { posA, posB, gridSnapshot, mergeSnapshot }
        this.startTime = null;
        this.timerInterval = null;
        this.solved = false;

        // 融合组: cellId -> groupId (Union-Find on cell indices)
        this.uf = new UnionFind(0);
        this.cellToGroup = {};   // cellId -> groupId

        // 拖拽状态
        this.dragStart = null;   // { row, col } 拖拽起点
        this.dragPos = null;     // 当前鼠标位置 { x, y } (canvas坐标)
        this.isDragging = false;
        this.dragGroup = null;   // 被拖拽的组的所有cellId列表
        this.dragGroupRect = null; // 被拖拽组的包围盒 { r, c, h, w }

        this._bindEvents();
    }

    // === 图片加载与切分 ===

    loadImage(img) {
        const { rows, cols } = this;
        const patchH = Math.floor(img.height / rows);
        const patchW = Math.floor(img.width / cols);

        this.canvas.width = patchW * cols;
        this.canvas.height = patchH * rows;
        this.patchW = patchW;
        this.patchH = patchH;

        this.pieces = [];
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = patchW;
        tempCanvas.height = patchH;
        const tempCtx = tempCanvas.getContext('2d');

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                tempCtx.clearRect(0, 0, patchW, patchH);
                tempCtx.drawImage(img, c * patchW, r * patchH, patchW, patchH, 0, 0, patchW, patchH);
                this.pieces.push({
                    image: tempCtx.getImageData(0, 0, patchW, patchH),
                    originalRow: r,
                    originalCol: c,
                });
            }
        }

        this._resetGrid();
        this._rebuildMerges();
        this.render();
    }

    _resetGrid() {
        const { rows, cols } = this;
        this.grid = [];
        for (let r = 0; r < rows; r++) {
            this.grid[r] = [];
            for (let c = 0; c < cols; c++) {
                this.grid[r][c] = r * cols + c;
            }
        }
    }

    // === 融合逻辑 ===

    _cellId(r, c) { return r * this.cols + c; }
    _cellRC(id) { return [Math.floor(id / this.cols), id % this.cols]; }

    _rebuildMerges() {
        const { rows, cols, grid, pieces } = this;
        const n = rows * cols;
        this.uf = new UnionFind(n);

        // 检查所有相邻对，如果都在正确位置且原图中也相邻则合并
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const piece = pieces[grid[r][c]];
                if (piece.originalRow !== r || piece.originalCol !== c) continue;

                // 右邻居
                if (c + 1 < cols) {
                    const rightPiece = pieces[grid[r][c + 1]];
                    if (rightPiece.originalRow === r && rightPiece.originalCol === c + 1) {
                        this.uf.union(this._cellId(r, c), this._cellId(r, c + 1));
                    }
                }
                // 下邻居
                if (r + 1 < rows) {
                    const downPiece = pieces[grid[r + 1][c]];
                    if (downPiece.originalRow === r + 1 && downPiece.originalCol === c) {
                        this.uf.union(this._cellId(r, c), this._cellId(r + 1, c));
                    }
                }
            }
        }

        // 更新 cellToGroup
        this.cellToGroup = {};
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const id = this._cellId(r, c);
                this.cellToGroup[id] = this.uf.find(id);
            }
        }
    }

    _getGroup(r, c) {
        const id = this._cellId(r, c);
        const root = this.uf.find(id);
        const cells = [];
        for (let rr = 0; rr < this.rows; rr++) {
            for (let cc = 0; cc < this.cols; cc++) {
                if (this.uf.find(this._cellId(rr, cc)) === root) {
                    cells.push(this._cellId(rr, cc));
                }
            }
        }
        return cells;
    }

    _getGroupBBox(cells) {
        let minR = Infinity, minC = Infinity, maxR = -1, maxC = -1;
        for (const id of cells) {
            const [r, c] = this._cellRC(id);
            minR = Math.min(minR, r); minC = Math.min(minC, c);
            maxR = Math.max(maxR, r); maxC = Math.max(maxC, c);
        }
        return { r: minR, c: minC, h: maxR - minR + 1, w: maxC - minC + 1 };
    }

    // === 游戏逻辑 ===

    shuffle(seed = 42) {
        const n = this.rows * this.cols;
        const arr = Array.from({ length: n }, (_, i) => i);

        let s = seed;
        function rand() {
            s |= 0; s = s + 0x6D2B79F5 | 0;
            let t = Math.imul(s ^ s >>> 15, 1 | s);
            t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
            return ((t ^ t >>> 14) >>> 0) / 4294967296;
        }

        for (let i = n - 1; i > 0; i--) {
            const j = Math.floor(rand() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]];
        }

        let idx = 0;
        for (let r = 0; r < this.rows; r++) {
            for (let c = 0; c < this.cols; c++) {
                this.grid[r][c] = arr[idx++];
            }
        }

        this.moveCount = 0;
        this.history = [];
        this.solved = false;
        this.startTime = Date.now();
        this._startTimer();
        this._rebuildMerges();
        this.onStateChange();
        this.render();
    }

    // 尝试交换两个组（多格 <-> 多格）
    _trySwapGroups(srcCells, targetR, targetC) {
        const srcBBox = this._getGroupBBox(srcCells);
        const srcCellSet = new Set(srcCells);

        // 收集目标区域所有cell (从targetR,targetC开始，按srcBBox的h×w扩展)
        const tgtCells = [];
        for (let dr = 0; dr < srcBBox.h; dr++) {
            for (let dc = 0; dc < srcBBox.w; dc++) {
                const tr = targetR + dr, tc = targetC + dc;
                if (tr < 0 || tr >= this.rows || tc < 0 || tc >= this.cols) return false;
                const tid = this._cellId(tr, tc);
                if (srcCellSet.has(tid)) return false; // 目标区域与源重叠
                tgtCells.push(tid);
            }
        }

        // 收集源区域cell (按BBox内位置顺序)
        const srcOrdered = [];
        for (let dr = 0; dr < srcBBox.h; dr++) {
            for (let dc = 0; dc < srcBBox.w; dc++) {
                srcOrdered.push(this._cellId(srcBBox.r + dr, srcBBox.c + dc));
            }
        }

        if (srcOrdered.length !== tgtCells.length) return false;

        // 保存快照用于撤销
        const gridSnap = this.grid.map(row => [...row]);
        const mergeSnap = { ...this.cellToGroup };

        // 执行交换: srcOrdered[i] <-> tgtCells[i]
        for (let i = 0; i < srcOrdered.length; i++) {
            const [sr, sc] = this._cellRC(srcOrdered[i]);
            const [tr, tc] = this._cellRC(tgtCells[i]);
            const tmp = this.grid[sr][sc];
            this.grid[sr][sc] = this.grid[tr][tc];
            this.grid[tr][tc] = tmp;
        }

        this.history.push({ gridSnapshot: gridSnap, mergeSnapshot: mergeSnap });
        this.moveCount++;
        this._rebuildMerges();
        this.onStateChange();
        this.render();

        if (this.isComplete()) {
            this.solved = true;
            this._stopTimer();
            this.onStateChange();
        }
        return true;
    }

    // 单格交换（两组大小不匹配时的降级方案）
    _swapSingle(r1, c1, r2, c2) {
        if (r1 === r2 && c1 === c2) return false;
        const gridSnap = this.grid.map(row => [...row]);
        const mergeSnap = { ...this.cellToGroup };

        const tmp = this.grid[r1][c1];
        this.grid[r1][c1] = this.grid[r2][c2];
        this.grid[r2][c2] = tmp;

        this.history.push({ gridSnapshot: gridSnap, mergeSnapshot: mergeSnap });
        this.moveCount++;
        this._rebuildMerges();
        this.onStateChange();
        this.render();

        if (this.isComplete()) {
            this.solved = true;
            this._stopTimer();
            this.onStateChange();
        }
        return true;
    }

    // 交换: 拖拽源组到目标位置
    swapGroupToTarget(srcCells, targetR, targetC) {
        if (this.solved) return;
        const srcBBox = this._getGroupBBox(srcCells);
        const srcSize = srcCells.length;

        // 检查目标区域大小
        const tgtCells = [];
        const srcCellSet = new Set(srcCells);
        for (let dr = 0; dr < srcBBox.h; dr++) {
            for (let dc = 0; dc < srcBBox.w; dc++) {
                const tr = targetR + dr, tc = targetC + dc;
                if (tr < 0 || tr >= this.rows || tc < 0 || tc >= this.cols) return;
                const tid = this._cellId(tr, tc);
                if (srcCellSet.has(tid)) return;
                tgtCells.push(tid);
            }
        }

        if (tgtCells.length === srcSize) {
            this._trySwapGroups(srcCells, targetR, targetC);
        } else if (srcSize === 1) {
            // 源是单格，目标也是单格
            const [sr, sc] = this._cellRC(srcCells[0]);
            this._swapSingle(sr, sc, targetR, targetC);
        }
        // 其他大小不匹配的情况不做交换
    }

    swapPieces(posA, posB) {
        if (this.solved) return;
        const [r1, c1] = posA;
        const [r2, c2] = posB;
        if (r1 === r2 && c1 === c2) return;

        const srcCells = this._getGroup(r1, c1);
        if (srcCells.length > 1) {
            this.swapGroupToTarget(srcCells, r2, c2);
        } else {
            const tgtCells = this._getGroup(r2, c2);
            if (tgtCells.length > 1) {
                this.swapGroupToTarget(tgtCells, r1, c1);
            } else {
                this._swapSingle(r1, c1, r2, c2);
            }
        }
    }

    undo() {
        if (this.history.length === 0 || this.solved) return;
        const last = this.history.pop();
        this.grid = last.gridSnapshot;
        this.cellToGroup = last.mergeSnapshot;
        // 重建UF from cellToGroup
        const n = this.rows * this.cols;
        this.uf = new UnionFind(n);
        // 用cellToGroup恢复: 同组的cell union在一起
        const groupMap = {};
        for (let id = 0; id < n; id++) {
            const gid = this.cellToGroup[id];
            if (gid !== undefined && gid !== id) {
                this.uf.union(id, gid);
            }
        }
        this.moveCount = Math.max(0, this.moveCount - 1);
        this.onStateChange();
        this.render();
    }

    isComplete() {
        for (let r = 0; r < this.rows; r++) {
            for (let c = 0; c < this.cols; c++) {
                const piece = this.pieces[this.grid[r][c]];
                if (piece.originalRow !== r || piece.originalCol !== c) return false;
            }
        }
        return true;
    }

    // === 状态API ===

    getGrid() { return this.grid.map(row => [...row]); }

    setGrid(newGrid) {
        this.grid = newGrid.map(row => [...row]);
        this._rebuildMerges();
        this.onStateChange();
        this.render();
    }

    getState() {
        return {
            grid: this.getGrid(), rows: this.rows, cols: this.cols,
            moveCount: this.moveCount, isComplete: this.isComplete(),
            accuracy: this.getPositionAccuracy(),
        };
    }

    getPositionAccuracy() {
        let correct = 0;
        const total = this.rows * this.cols;
        for (let r = 0; r < this.rows; r++) {
            for (let c = 0; c < this.cols; c++) {
                const piece = this.pieces[this.grid[r][c]];
                if (piece.originalRow === r && piece.originalCol === c) correct++;
            }
        }
        return correct / total;
    }

    // === 渲染 ===

    render() {
        const { ctx, canvas, rows, cols, patchW, patchH, pieces, grid } = this;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // 1. 绘制所有拼图块
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                this._drawPiece(r, c, false);
            }
        }

        // 2. 绘制融合组边框（隐藏组内网格线，绘制组外框）
        this._drawMergeBorders();

        // 3. 绘制拖拽中的组（半透明）
        if (this.isDragging && this.dragGroup && this.dragPos) {
            this._drawDragGhost();
        }
    }

    _drawPiece(r, c, isGhost) {
        const { ctx, patchW, patchH, pieces, grid } = this;
        const pieceIdx = grid[r][c];
        const piece = pieces[pieceIdx];
        const x = c * patchW;
        const y = r * patchH;

        const tmpCanvas = document.createElement('canvas');
        tmpCanvas.width = patchW;
        tmpCanvas.height = patchH;
        tmpCanvas.getContext('2d').putImageData(piece.image, 0, 0);

        if (isGhost) {
            ctx.globalAlpha = 0.5;
        }
        ctx.drawImage(tmpCanvas, x, y);
        ctx.globalAlpha = 1.0;

        // 正确位置绿点
        if (piece.originalRow === r && piece.originalCol === c) {
            ctx.fillStyle = 'rgba(83, 215, 105, 0.7)';
            ctx.beginPath();
            ctx.arc(x + 12, y + 12, 6, 0, Math.PI * 2);
            ctx.fill();
        }

        // 网格线
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, patchW, patchH);
    }

    _drawMergeBorders() {
        const { ctx, rows, cols, patchW, patchH } = this;
        const drawn = new Set();

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const id = this._cellId(r, c);
                const gid = this.uf.find(id);
                if (drawn.has(gid)) continue;
                drawn.add(gid);

                const cells = this._getGroup(r, c);
                if (cells.length <= 1) continue;

                // 绘制组外边框
                const bbox = this._getGroupBBox(cells);
                const x = bbox.c * patchW;
                const y = bbox.r * patchH;
                const w = bbox.w * patchW;
                const h = bbox.h * patchH;

                ctx.strokeStyle = 'rgba(83, 215, 105, 0.9)';
                ctx.lineWidth = 3;
                ctx.strokeRect(x + 1, y + 1, w - 2, h - 2);

                // 隐藏组内网格线：在组内相邻边界画背景色线
                ctx.strokeStyle = 'rgba(26, 26, 46, 0.8)'; // 背景色
                ctx.lineWidth = 2;
                for (const cid of cells) {
                    const [cr, cc] = this._cellRC(cid);
                    // 检查右邻居是否同组
                    if (cc + 1 < cols && this.uf.find(this._cellId(cr, cc + 1)) === gid) {
                        ctx.beginPath();
                        ctx.moveTo((cc + 1) * patchW, cr * patchH);
                        ctx.lineTo((cc + 1) * patchW, (cr + 1) * patchH);
                        ctx.stroke();
                    }
                    // 检查下邻居是否同组
                    if (cr + 1 < rows && this.uf.find(this._cellId(cr + 1, cc)) === gid) {
                        ctx.beginPath();
                        ctx.moveTo(cc * patchW, (cr + 1) * patchH);
                        ctx.lineTo((cc + 1) * patchW, (cr + 1) * patchH);
                        ctx.stroke();
                    }
                }
            }
        }
    }

    _drawDragGhost() {
        const { ctx, patchW, patchH, dragGroup, dragPos, dragStart } = this;
        const bbox = this._getGroupBBox(dragGroup);

        // 计算拖拽偏移（鼠标当前位置 - 拖拽起点）
        const startX = dragStart.col * patchW + patchW / 2;
        const startY = dragStart.row * patchH + patchH / 2;
        const dx = dragPos.x - startX;
        const dy = dragPos.y - startY;

        ctx.save();
        ctx.globalAlpha = 0.4;

        // 在原位画半透明（表示离开的位置）
        for (const id of dragGroup) {
            const [r, c] = this._cellRC(id);
            const pieceIdx = this.grid[r][c];
            const piece = this.pieces[pieceIdx];
            const tmpCanvas = document.createElement('canvas');
            tmpCanvas.width = patchW;
            tmpCanvas.height = patchH;
            tmpCanvas.getContext('2d').putImageData(piece.image, 0, 0);
            ctx.drawImage(tmpCanvas, c * patchW, r * patchH);
        }

        // 在鼠标位置画拖拽中的组
        ctx.globalAlpha = 0.7;
        const ox = bbox.c * patchW + dx;
        const oy = bbox.r * patchH + dy;
        for (const id of dragGroup) {
            const [r, c] = this._cellRC(id);
            const pieceIdx = this.grid[r][c];
            const piece = this.pieces[pieceIdx];
            const tmpCanvas = document.createElement('canvas');
            tmpCanvas.width = patchW;
            tmpCanvas.height = patchH;
            tmpCanvas.getContext('2d').putImageData(piece.image, 0, 0);
            const px = ox + (c - bbox.c) * patchW;
            const py = oy + (r - bbox.r) * patchH;
            ctx.drawImage(tmpCanvas, px, py);
        }

        // 拖拽组边框
        ctx.strokeStyle = '#4a90d9';
        ctx.lineWidth = 3;
        ctx.strokeRect(ox, oy, bbox.w * patchW, bbox.h * patchH);

        ctx.restore();
    }

    // === 事件处理 ===

    _bindEvents() {
        this.canvas.addEventListener('mousedown', (e) => this._onPointerDown(e));
        this.canvas.addEventListener('mousemove', (e) => this._onPointerMove(e));
        this.canvas.addEventListener('mouseup', (e) => this._onPointerUp(e));
        this.canvas.addEventListener('touchstart', (e) => { e.preventDefault(); this._onPointerDown(e.touches[0]); }, { passive: false });
        this.canvas.addEventListener('touchmove', (e) => { e.preventDefault(); this._onPointerMove(e.touches[0]); }, { passive: false });
        this.canvas.addEventListener('touchend', (e) => { e.preventDefault(); this._onPointerUp(e.changedTouches[0]); }, { passive: false });
    }

    _getCanvasPos(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY,
        };
    }

    _getGridPos(e) {
        const pos = this._getCanvasPos(e);
        const col = Math.floor(pos.x / this.patchW);
        const row = Math.floor(pos.y / this.patchH);
        if (row < 0 || row >= this.rows || col < 0 || col >= this.cols) return null;
        return { row, col };
    }

    _onPointerDown(e) {
        const pos = this._getGridPos(e);
        if (!pos) return;

        this.dragStart = pos;
        this.isDragging = false;
        this.dragGroup = this._getGroup(pos.row, pos.col);
        this.dragGroupRect = this._getGroupBBox(this.dragGroup);
    }

    _onPointerMove(e) {
        if (!this.dragStart) return;
        this.isDragging = true;
        this.dragPos = this._getCanvasPos(e);
        this.render();
    }

    _onPointerUp(e) {
        if (this.dragStart && this.isDragging) {
            const pos = this._getGridPos(e);
            if (pos) {
                const bbox = this.dragGroupRect;
                const isSameGroup = this.dragGroup.includes(this._cellId(pos.row, pos.col));

                if (!isSameGroup && (pos.row !== this.dragStart.row || pos.col !== this.dragStart.col)) {
                    // 目标位置作为新区域的左上角
                    this.swapGroupToTarget(this.dragGroup, pos.row, pos.col);
                }
            }
        }
        this.dragStart = null;
        this.isDragging = false;
        this.dragPos = null;
        this.dragGroup = null;
        this.dragGroupRect = null;
        this.render();
    }

    // === 计时器 ===

    _startTimer() {
        this._stopTimer();
        this.timerInterval = setInterval(() => {
            const elapsed = Math.floor((Date.now() - this.startTime) / 1000);
            const min = String(Math.floor(elapsed / 60)).padStart(2, '0');
            const sec = String(elapsed % 60).padStart(2, '0');
            document.getElementById('timer').textContent = `${min}:${sec}`;
        }, 1000);
    }

    _stopTimer() {
        if (this.timerInterval) { clearInterval(this.timerInterval); this.timerInterval = null; }
    }
}

// === PuzzleAPI ===
class PuzzleAPI {
    constructor(game) { this.game = game; }
    getState() { return this.game.getState(); }
    getGrid() { return this.game.getGrid(); }
    setGrid(grid) { this.game.setGrid(grid); }
    swap(a, b) {
        this.game.swapPieces(a, b);
        return { done: this.game.isComplete(), accuracy: this.game.getPositionAccuracy() };
    }
    reset(seed) { this.game.shuffle(seed || 42); }
    isComplete() { return this.game.isComplete(); }
    getAccuracy() { return this.game.getPositionAccuracy(); }
}

// === 初始化 ===
document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('puzzleCanvas');
    const game = new PuzzleGame(canvas, { rows: 6, cols: 6, onStateChange: updateStats });
    const api = new PuzzleAPI(game);
    window.puzzleAPI = api;

    function updateStats() {
        document.getElementById('moveCount').textContent = game.moveCount;
        document.getElementById('accuracy').textContent = (game.getPositionAccuracy() * 100).toFixed(1) + '%';
        const msgEl = document.getElementById('message');
        if (game.solved) {
            msgEl.textContent = '恭喜完成拼图!';
            msgEl.className = 'message';
        } else {
            msgEl.textContent = '';
            msgEl.className = 'message';
        }
    }

    document.getElementById('btnNewGame').addEventListener('click', () => {
        const size = parseInt(document.getElementById('gridSize').value);
        game.rows = size;
        game.cols = size;
        fetch(`/api/image?size=${size * 100}&seed=${Date.now()}`)
            .then(r => r.json())
            .then(data => {
                const img = new Image();
                img.onload = () => {
                    game.loadImage(img);
                    game.shuffle(Date.now() % 10000);
                    updateStats();
                };
                img.src = 'data:image/png;base64,' + data.image;
            })
            .catch(() => loadDefaultImage(size));
    });

    document.getElementById('btnUpload').addEventListener('click', () => document.getElementById('fileInput').click());
    document.getElementById('fileInput').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            const img = new Image();
            img.onload = () => {
                const size = parseInt(document.getElementById('gridSize').value);
                game.rows = size;
                game.cols = size;
                game.loadImage(img);
                game.shuffle(Date.now() % 10000);
                updateStats();
            };
            img.src = ev.target.result;
        };
        reader.readAsDataURL(file);
    });

    document.getElementById('btnReset').addEventListener('click', () => {
        if (game.pieces.length > 0) { game.shuffle(game.moveCount || 42); updateStats(); }
    });

    document.getElementById('btnUndo').addEventListener('click', () => { game.undo(); updateStats(); });

    function loadDefaultImage(size) {
        const c = document.createElement('canvas');
        c.width = size * 100;
        c.height = size * 100;
        const ctx = c.getContext('2d');
        for (let r = 0; r < size; r++) {
            for (let col = 0; col < size; col++) {
                const hue = (r * size + col) * (360 / (size * size));
                ctx.fillStyle = `hsl(${hue}, 70%, 60%)`;
                ctx.fillRect(col * 100, r * 100, 100, 100);
                ctx.fillStyle = 'white';
                ctx.font = 'bold 24px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(`${r},${col}`, col * 100 + 50, r * 100 + 50);
            }
        }
        const img = new Image();
        img.onload = () => { game.loadImage(img); game.shuffle(Date.now() % 10000); updateStats(); };
        img.src = c.toDataURL();
    }

    document.getElementById('btnNewGame').click();
});
