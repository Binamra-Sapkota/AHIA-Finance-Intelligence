"""Book Library View for AHIA-R2.

Features:
- Document list table (Title, Author/Source, Format, Chunks, Words, Status)
- Add Document button supporting PDF, TXT, and Markdown files
- Document metadata inspector panel
- Index rebuild controls and progress tracking
"""

from pathlib import Path
from typing import Dict, List, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class LibraryView(QWidget):
    """Book library management view showing catalog, ingestion status, and metadata."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.documents: List[Dict[str, str]] = []
        self._setup_ui()
        self._load_demo_library()

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Header Bar with Controls
        header_card = QFrame()
        header_card.setProperty("class", "card")
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(12, 10, 12, 10)

        title_layout = QVBoxLayout()
        view_title = QLabel("Personal Finance Knowledge Library")
        view_title.setStyleSheet("font-size: 15px; font-weight: 600; color: #FFFFFF;")

        view_subtitle = QLabel("Curated local corpus indexed via Okapi BM25 and SQLite")
        view_subtitle.setStyleSheet("font-size: 11px; color: #94A3B8;")

        title_layout.addWidget(view_title)
        title_layout.addWidget(view_subtitle)
        header_layout.addLayout(title_layout)

        header_layout.addStretch()

        demo_badge = QLabel("DEMONSTRATION CATALOG")
        demo_badge.setProperty("class", "badge-demo")
        header_layout.addWidget(demo_badge)

        self.rebuild_btn = QPushButton("🔄 Rebuild BM25 Index")
        self.rebuild_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.rebuild_btn.clicked.connect(self._on_rebuild_index)
        header_layout.addWidget(self.rebuild_btn)

        self.add_btn = QPushButton("➕ Add Document")
        self.add_btn.setProperty("class", "primary")
        self.add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.add_btn.setToolTip("Add a PDF, TXT or Markdown document to local library")
        self.add_btn.clicked.connect(self._on_add_document)
        header_layout.addWidget(self.add_btn)

        main_layout.addWidget(header_card)

        # Indexing Progress Bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        main_layout.addWidget(self.progress_bar)

        # Splitter: Table on left, Metadata Inspector on right
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #253043; width: 2px; }")

        # --- Table Widget ---
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 8, 0)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Title", "Author / Source", "Format", "Words", "Chunks", "Indexing Status"
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        table_layout.addWidget(self.table)

        splitter.addWidget(table_container)

        # --- Metadata Inspector Panel ---
        inspector_card = QFrame()
        inspector_card.setProperty("class", "card")
        inspector_layout = QVBoxLayout(inspector_card)
        inspector_layout.setContentsMargins(14, 14, 14, 14)
        inspector_layout.setSpacing(10)

        inspector_title = QLabel("Document Metadata Inspector")
        inspector_title.setStyleSheet("font-size: 13px; font-weight: 600; color: #60A5FA;")
        inspector_layout.addWidget(inspector_title)

        self.lbl_meta_title = QLabel("Select a book from the list to inspect metadata.")
        self.lbl_meta_title.setStyleSheet("font-weight: 600; font-size: 13px; color: #FFFFFF;")
        self.lbl_meta_title.setWordWrap(True)
        inspector_layout.addWidget(self.lbl_meta_title)

        self.lbl_meta_author = QLabel("Author: —")
        self.lbl_meta_format = QLabel("Format: —")
        self.lbl_meta_path = QLabel("Local Path: —")
        self.lbl_meta_path.setWordWrap(True)
        self.lbl_meta_words = QLabel("Total Word Count: —")
        self.lbl_meta_chunks = QLabel("Generated Chunks: —")
        self.lbl_meta_status = QLabel("Status: —")

        for lbl in [
            self.lbl_meta_author,
            self.lbl_meta_format,
            self.lbl_meta_path,
            self.lbl_meta_words,
            self.lbl_meta_chunks,
            self.lbl_meta_status,
        ]:
            lbl.setStyleSheet("font-size: 12px; color: #94A3B8; font-family: monospace;")
            inspector_layout.addWidget(lbl)

        inspector_layout.addStretch()

        # Storage Note
        note_box = QFrame()
        note_box.setStyleSheet("background-color: #0E131E; border: 1px solid #1F2837; border-radius: 6px; padding: 8px;")
        note_layout = QVBoxLayout(note_box)
        note_layout.setContentsMargins(6, 6, 6, 6)
        note_lbl = QLabel("Private Storage Notice")
        note_lbl.setStyleSheet("color: #F59E0B; font-weight: 600; font-size: 10px; text-transform: uppercase;")
        note_desc = QLabel("Raw books remain 100% on local disk (data/raw/). Chunks are stored in local SQLite database.")
        note_desc.setWordWrap(True)
        note_desc.setStyleSheet("color: #64748B; font-size: 10px;")
        note_layout.addWidget(note_lbl)
        note_layout.addWidget(note_desc)
        inspector_layout.addWidget(note_box)

        splitter.addWidget(inspector_card)
        splitter.setSizes([640, 320])

        main_layout.addWidget(splitter)

    def _load_demo_library(self) -> None:
        """Populate initial library with personal-finance books."""
        self.documents = [
            {
                "title": "The Psychology of Money",
                "author": "Morgan Housel",
                "format": "PDF",
                "words": "62,400",
                "chunks": "248",
                "status": "Indexed (BM25)",
                "path": "Books PDF/The Psychology of Money.pdf",
            },
            {
                "title": "The Richest Man in Babylon",
                "author": "George S. Clason",
                "format": "PDF",
                "words": "41,800",
                "chunks": "167",
                "status": "Indexed (BM25)",
                "path": "data/books/Richest_Man_Babylon.pdf",
            },
            {
                "title": "Think & Grow Rich",
                "author": "Napoleon Hill",
                "format": "PDF",
                "words": "104,200",
                "chunks": "416",
                "status": "Indexed (BM25)",
                "path": "Books PDF/Think & Grow Rich.pdf",
            },
            {
                "title": "Rich Dad Poor Dad",
                "author": "Robert T. Kiyosaki",
                "format": "PDF",
                "words": "58,900",
                "chunks": "235",
                "status": "Indexed (BM25)",
                "path": "Books PDF/Rich Dad Poor Dad.pdf",
            },
            {
                "title": "The Millionaire Next Door",
                "author": "Thomas J. Stanley & William D. Danko",
                "format": "PDF",
                "words": "82,500",
                "chunks": "330",
                "status": "Indexed (BM25)",
                "path": "Books PDF/The Millionaire Next Door.pdf",
            },
            {
                "title": "Your Money Your Life",
                "author": "Vicki Robin & Joe Dominguez",
                "format": "PDF",
                "words": "94,100",
                "chunks": "376",
                "status": "Indexed (BM25)",
                "path": "Books PDF/Your Money Your Life.pdf",
            },
            {
                "title": "Atomic Habits",
                "author": "James Clear",
                "format": "PDF",
                "words": "75,300",
                "chunks": "301",
                "status": "Indexed (BM25)",
                "path": "Books PDF/Atomic Habits.pdf",
            },
        ]
        self._refresh_table()
        if self.table.rowCount() > 0:
            self.table.selectRow(0)

    def _refresh_table(self) -> None:
        self.table.setRowCount(len(self.documents))
        for row, doc in enumerate(self.documents):
            item_title = QTableWidgetItem(doc["title"])
            item_author = QTableWidgetItem(doc["author"])
            item_format = QTableWidgetItem(doc["format"])
            item_words = QTableWidgetItem(doc["words"])
            item_chunks = QTableWidgetItem(doc["chunks"])
            item_status = QTableWidgetItem(f"● {doc['status']}")

            # Text alignment
            item_format.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_words.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            item_chunks.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

            # Styling status
            item_status.setForeground(Qt.GlobalColor.green)

            self.table.setItem(row, 0, item_title)
            self.table.setItem(row, 1, item_author)
            self.table.setItem(row, 2, item_format)
            self.table.setItem(row, 3, item_words)
            self.table.setItem(row, 4, item_chunks)
            self.table.setItem(row, 5, item_status)

    def _on_row_selected(self) -> None:
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        if 0 <= row < len(self.documents):
            doc = self.documents[row]
            self.lbl_meta_title.setText(doc["title"])
            self.lbl_meta_author.setText(f"Author: {doc['author']}")
            self.lbl_meta_format.setText(f"Format: {doc['format']}")
            self.lbl_meta_path.setText(f"Local Path: {doc.get('path', 'N/A')}")
            self.lbl_meta_words.setText(f"Total Word Count: {doc['words']} words")
            self.lbl_meta_chunks.setText(f"Generated Chunks: {doc['chunks']} passages (250w window)")
            self.lbl_meta_status.setText(f"Status: {doc['status']}")

    def _on_add_document(self) -> None:
        """Open file dialog for PDF, TXT, or Markdown documents."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Personal-Finance Document",
            "",
            "Supported Documents (*.pdf *.txt *.md);;PDF Files (*.pdf);;Text Files (*.txt);;Markdown Files (*.md)",
        )
        if file_path:
            p = Path(file_path)
            new_doc = {
                "title": p.stem.replace("_", " ").title(),
                "author": "Local User Document",
                "format": p.suffix.upper().replace(".", ""),
                "words": "Estimating...",
                "chunks": "Pending...",
                "status": "Ready to Index",
                "path": str(p),
            }
            self.documents.append(new_doc)
            self._refresh_table()
            self.table.selectRow(len(self.documents) - 1)

    def _on_rebuild_index(self) -> None:
        """Simulate BM25 index rebuilding for demonstration."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.rebuild_btn.setEnabled(False)
        self.rebuild_btn.setText("Indexing Corpus...")

        # Small step updates
        from PySide6.QtCore import QTimer
        def step(val: int) -> None:
            self.progress_bar.setValue(val)
            if val >= 100:
                self.progress_bar.setVisible(False)
                self.rebuild_btn.setEnabled(True)
                self.rebuild_btn.setText("🔄 Rebuild BM25 Index")
            else:
                QTimer.singleShot(150, lambda: step(val + 25))

        QTimer.singleShot(100, lambda: step(25))
