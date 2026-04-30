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
        this._pieceCanvasCache = [];
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = patchW;
        tempCanvas.height = patchH;
        const tempCtx = tempCanvas.getContext('2d');

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                tempCtx.clearRect(0, 0, patchW, patchH);
                tempCtx.drawImage(img, c * patchW, r * patchH, patchW, patchH, 0, 0, patchW, patchH);
                const imgData = tempCtx.getImageData(0, 0, patchW, patchH);
                this.pieces.push({
                    image: imgData,
                    originalRow: r,
                    originalCol: c,
                });
                // 预缓存离屏canvas
                const cached = document.createElement('canvas');
                cached.width = patchW;
                cached.height = patchH;
                cached.getContext('2d').putImageData(imgData, 0, 0);
                this._pieceCanvasCache.push(cached);
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
        const oldUf = this.uf;

        this.uf = new UnionFind(n);

        // 合并条件：网格中相邻 + 原图中也相邻（同方向），不要求在绝对正确位置
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const piece = pieces[grid[r][c]];

                // 右邻居：当前在左边，原图中也应在左边（originalCol 差 1）
                if (c + 1 < cols) {
                    const rightPiece = pieces[grid[r][c + 1]];
                    if (piece.originalRow === rightPiece.originalRow &&
                        piece.originalCol - rightPiece.originalCol === -1) {
                        this.uf.union(this._cellId(r, c), this._cellId(r, c + 1));
                    }
                }
                // 下邻居：当前在上边，原图中也应在上边（originalRow 差 1）
                if (r + 1 < rows) {
                    const downPiece = pieces[grid[r + 1][c]];
                    if (piece.originalCol === downPiece.originalCol &&
                        piece.originalRow - downPiece.originalRow === -1) {
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

        // 只要有合并组（size > 1）就触发融合动画
        const roots = new Set();
        for (let i = 0; i < n; i++) roots.add(this.uf.find(i));
        if (roots.size < n) {
            this._animateMerge();
        }
    }

    _animateMerge() {
        const duration = 1000;
        const start = performance.now();

        const animate = (now) => {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            this._mergeAnimProgress = progress;
            this.render();
            if (progress < 1) {
                requestAnimationFrame(animate);
            } else {
                this._mergeAnimProgress = null;
            }
        };
        requestAnimationFrame(animate);
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

    // 交换组内实际cell到目标位置（anchorR,anchorC为点击的cell，作为偏移锚点）
    // 使用临时缓冲区：先保存所有源位置的值，再写入目标，最后将被置换块填入空出的源位置
    _trySwapGroups(srcCells, anchorR, anchorC, targetR, targetC) {
        const srcCellSet = new Set(srcCells);

        // 以点击cell为锚点计算每个cell的目标位置
        const moves = []; // { srcId, tgtId }
        for (const id of srcCells) {
            const [r, c] = this._cellRC(id);
            const dr = r - anchorR, dc = c - anchorC;
            const tr = targetR + dr, tc = targetC + dc;
            if (tr < 0 || tr >= this.rows || tc < 0 || tc >= this.cols) return false;
            const tid = this._cellId(tr, tc);
            if (tid === id) continue; // 自身位置，跳过
            moves.push({ srcId: id, tgtId: tid });
        }

        // 保存快照用于撤销
        const gridSnap = this.grid.map(row => [...row]);
        const mergeSnap = { ...this.cellToGroup };

        // 1. 用临时缓冲区保存所有源位置的值
        const savedValues = new Map();
        for (const { srcId } of moves) {
            const [sr, sc] = this._cellRC(srcId);
            savedValues.set(srcId, this.grid[sr][sc]);
        }

        // 2. 收集被置换块（目标位置不在融合块范围内的块）
        const displacedPieces = [];
        for (const { tgtId } of moves) {
            if (!srcCellSet.has(tgtId)) {
                const [tr, tc] = this._cellRC(tgtId);
                displacedPieces.push(this.grid[tr][tc]);
            }
        }

        // 3. 将源块写入目标位置
        for (const { srcId, tgtId } of moves) {
            const [tr, tc] = this._cellRC(tgtId);
            this.grid[tr][tc] = savedValues.get(srcId);
        }

        // 4. 将被置换块填入空出的源位置
        //    条件：该源位置没有被其他组成员的目标位置覆盖
        //    即没有其他move的tgtId等于当前srcId
        const filledByGroup = new Set(moves.map(m => m.tgtId));
        let dispIdx = 0;
        for (const { srcId } of moves) {
            if (!filledByGroup.has(srcId) && dispIdx < displacedPieces.length) {
                const [sr, sc] = this._cellRC(srcId);
                this.grid[sr][sc] = displacedPieces[dispIdx];
                dispIdx++;
            }
        }

        this._rebuildMerges();

        this.history.push({ gridSnapshot: gridSnap, mergeSnapshot: mergeSnap });
        this.moveCount++;
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

    // 交换: 拖拽源组到目标位置，anchorR/anchorC为点击的cell
    // 融合块拖到单块: 整个融合块移到目标区域，单块被置换到源区域
    // 融合块拖到另一融合块: 逐块交换，目标块从其融合组中分割出来
    swapGroupToTarget(srcCells, anchorR, anchorC, targetR, targetC) {
        if (this.solved) return;
        if (srcCells.length === 1) {
            const [sr, sc] = this._cellRC(srcCells[0]);
            this._swapSingle(sr, sc, targetR, targetC);
        } else {
            this._trySwapGroups(srcCells, anchorR, anchorC, targetR, targetC);
        }
    }

    swapPieces(posA, posB) {
        if (this.solved) return;
        const [r1, c1] = posA;
        const [r2, c2] = posB;
        if (r1 === r2 && c1 === c2) return;

        const srcCells = this._getGroup(r1, c1);
        if (srcCells.length > 1) {
            this.swapGroupToTarget(srcCells, r1, c1, r2, c2);
        } else {
            const tgtCells = this._getGroup(r2, c2);
            if (tgtCells.length > 1) {
                this.swapGroupToTarget(tgtCells, r2, c2, r1, c1);
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
        const { rows, cols, pieces, grid } = this;
        let correct = 0;
        let total = 0;
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const p1 = pieces[grid[r][c]];
                // 右边
                if (c + 1 < cols) {
                    total++;
                    const p2 = pieces[grid[r][c + 1]];
                    if (p1.originalRow === p2.originalRow && Math.abs(p1.originalCol - p2.originalCol) === 1) correct++;
                }
                // 下边
                if (r + 1 < rows) {
                    total++;
                    const p2 = pieces[grid[r + 1][c]];
                    if (p1.originalCol === p2.originalCol && Math.abs(p1.originalRow - p2.originalRow) === 1) correct++;
                }
            }
        }
        return total > 0 ? correct / total : 0;
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

        if (isGhost) {
            ctx.globalAlpha = 0.5;
        }
        ctx.drawImage(this._pieceCanvasCache[pieceIdx], x, y);
        ctx.globalAlpha = 1.0;

        // 网格线
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, patchW, patchH);
    }

    _drawMergeBorders() {
        const { ctx, rows, cols, patchW, patchH, pieces, grid } = this;
        const animP = this._mergeAnimProgress; // null if not animating
        const alpha = animP !== null ? 0.9 * animP : 0.9;

        ctx.strokeStyle = `rgba(83, 215, 105, ${alpha})`;
        ctx.lineWidth = 3;

        // 判断两个cell中的piece在原图中是否相邻
        const _originallyAdjacent = (r1, c1, r2, c2) => {
            const p1 = pieces[grid[r1][c1]], p2 = pieces[grid[r2][c2]];
            if (r1 === r2 && Math.abs(c1 - c2) === 1) {
                return p1.originalRow === p2.originalRow && Math.abs(p1.originalCol - p2.originalCol) === 1;
            }
            if (c1 === c2 && Math.abs(r1 - r2) === 1) {
                return p1.originalCol === p2.originalCol && Math.abs(p1.originalRow - p2.originalRow) === 1;
            }
            return false;
        };

        // 遍历所有相邻边，原图中相邻的边不高亮，其余都高亮
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                // 右边
                if (c + 1 < cols && !_originallyAdjacent(r, c, r, c + 1)) {
                    const x = (c + 1) * patchW;
                    ctx.beginPath();
                    ctx.moveTo(x, r * patchH);
                    ctx.lineTo(x, (r + 1) * patchH);
                    ctx.stroke();
                }
                // 下边
                if (r + 1 < rows && !_originallyAdjacent(r, c, r + 1, c)) {
                    const y = (r + 1) * patchH;
                    ctx.beginPath();
                    ctx.moveTo(c * patchW, y);
                    ctx.lineTo((c + 1) * patchW, y);
                    ctx.stroke();
                }
            }
        }
    }

    _drawDragGhost() {
        const { ctx, patchW, patchH, dragGroup, dragPos, dragStart } = this;

        // 拖拽偏移：以点击的cell为锚点
        const anchorX = dragStart.col * patchW;
        const anchorY = dragStart.row * patchH;
        const dx = dragPos.x - anchorX - patchW / 2;
        const dy = dragPos.y - anchorY - patchH / 2;

        ctx.save();

        // 在原位画半透明（表示离开的位置）
        ctx.globalAlpha = 0.4;
        for (const id of dragGroup) {
            const [r, c] = this._cellRC(id);
            const pieceIdx = this.grid[r][c];
            ctx.drawImage(this._pieceCanvasCache[pieceIdx], c * patchW, r * patchH);
        }

        // 在鼠标位置画拖拽中的组（只画实际cell，不画bbox空洞）
        ctx.globalAlpha = 0.7;
        for (const id of dragGroup) {
            const [r, c] = this._cellRC(id);
            const pieceIdx = this.grid[r][c];
            const px = c * patchW + dx;
            const py = r * patchH + dy;
            ctx.drawImage(this._pieceCanvasCache[pieceIdx], px, py);

            // 每个cell的蓝色边框
            ctx.strokeStyle = '#4a90d9';
            ctx.lineWidth = 2;
            ctx.strokeRect(px, py, patchW, patchH);
        }

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
            if (pos && (pos.row !== this.dragStart.row || pos.col !== this.dragStart.col)) {
                // 以点击cell为锚点，目标位置为放下位置
                this.swapGroupToTarget(this.dragGroup, this.dragStart.row, this.dragStart.col, pos.row, pos.col);
            }
        }
        this.dragStart = null;
        this.isDragging = false;
        this.dragPos = null;
        this.dragGroup = null;
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
        if (game.pieces.length > 0) { game.shuffle(Date.now() % 10000); updateStats(); }
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
