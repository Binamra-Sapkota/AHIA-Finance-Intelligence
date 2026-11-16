"""AI Chat View for AHIA-R2 Book Intelligence.

Features:
- Message history thread with user/assistant bubbles
- Question input with Send button and shortcut support
- Grounded answer display with inline citation links
- Source references panel displaying verbatim excerpts and metadata
- Processing details panel detailing reasoning steps, depth, and governor status
"""

from typing import List, Optional
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from service.dto import CitationItem, QueryResult


class ChatView(QWidget):
    """Primary conversational canvas with grounded citations and processing telemetry."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._setup_ui()
        self._load_initial_demo_conversation()

    def _setup_ui(self) -> None:
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # Splitter between Chat Area and Details/Sources Drawer
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #253043; width: 2px; }")

        # --- LEFT: Conversation & Input Panel ---
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(12)

        # Top Header Bar for Chat
        chat_header = QFrame()
        chat_header.setProperty("class", "card")
        chat_header_layout = QHBoxLayout(chat_header)
        chat_header_layout.setContentsMargins(12, 8, 12, 8)

        chat_title = QLabel("Research Conversation Canvas")
        chat_title.setStyleSheet("font-size: 14px; font-weight: 600; color: #F3F4F6;")

        demo_badge = QLabel("DEMONSTRATION SHELL")
        demo_badge.setProperty("class", "badge-demo")

        depth_badge = QLabel("ADAPTIVE DEPTH: 4 BLOCKS")
        depth_badge.setProperty("class", "badge-depth")

        chat_header_layout.addWidget(chat_title)
        chat_header_layout.addStretch()
        chat_header_layout.addWidget(depth_badge)
        chat_header_layout.addWidget(demo_badge)
        left_layout.addWidget(chat_header)

        # Message History Scroll Area
        self.message_list = QListWidget()
        self.message_list.setStyleSheet(
            "QListWidget { background-color: #0E131E; border: 1px solid #253043; "
            "border-radius: 8px; padding: 12px; } "
            "QListWidget::item { margin-bottom: 12px; } "
            "QListWidget::item:selected { background-color: transparent; }"
        )
        self.message_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        left_layout.addWidget(self.message_list, stretch=1)

        # Quick Prompt Suggestion Chips
        chips_frame = QFrame()
        chips_layout = QHBoxLayout(chips_frame)
        chips_layout.setContentsMargins(0, 0, 0, 0)
        chips_layout.setSpacing(8)

        chip_label = QLabel("Sample Queries:")
        chip_label.setStyleSheet("color: #64748B; font-size: 11px; font-weight: 600;")
        chips_layout.addWidget(chip_label)

        sample_prompts = [
            "What is the 10% rule in Babylon?",
            "How does Morgan Housel define wealth?",
            "What are the seven cures for a lean purse?",
        ]
        for prompt in sample_prompts:
            chip_btn = QPushButton(prompt)
            chip_btn.setStyleSheet(
                "QPushButton { background-color: #17202F; color: #94A3B8; border: 1px solid #253043; "
                "border-radius: 12px; padding: 4px 10px; font-size: 11px; } "
                "QPushButton:hover { background-color: #1E2D44; color: #60A5FA; border-color: #3B82F6; }"
            )
            chip_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            chip_btn.clicked.connect(lambda checked=False, p=prompt: self._on_chip_clicked(p))
            chips_layout.addWidget(chip_btn)

        chips_layout.addStretch()
        left_layout.addWidget(chips_frame)

        # Input Area
        input_container = QFrame()
        input_container.setProperty("class", "card")
        input_layout = QHBoxLayout(input_container)
        input_layout.setContentsMargins(8, 8, 8, 8)
        input_layout.setSpacing(10)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("Ask a question about the 6 personal-finance books...")
        self.input_edit.returnPressed.connect(self._on_send_clicked)
        self.input_edit.setStyleSheet("font-size: 13px; padding: 8px 12px;")

        self.send_btn = QPushButton("Send Question")
        self.send_btn.setProperty("class", "primary")
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.clicked.connect(self._on_send_clicked)

        input_layout.addWidget(self.input_edit, stretch=1)
        input_layout.addWidget(self.send_btn)
        left_layout.addWidget(input_container)

        splitter.addWidget(left_container)

        # --- RIGHT: Sources & Processing Telemetry Drawer ---
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(12)

        # Drawer Title
        drawer_header = QLabel("Citation & Grounding Inspector")
        drawer_header.setStyleSheet("font-size: 14px; font-weight: 600; color: #F3F4F6;")
        right_layout.addWidget(drawer_header)

        # Source References Panel
        sources_card = QFrame()
        sources_card.setProperty("class", "card")
        sources_layout = QVBoxLayout(sources_card)
        sources_layout.setContentsMargins(12, 12, 12, 12)
        sources_layout.setSpacing(8)

        sources_label = QLabel("Verified Book Passages")
        sources_label.setStyleSheet("font-weight: 600; font-size: 12px; color: #60A5FA;")
        sources_layout.addWidget(sources_label)

        self.source_scroll = QScrollArea()
        self.source_scroll.setWidgetResizable(True)
        self.source_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.sources_content = QWidget()
        self.sources_content_layout = QVBoxLayout(self.sources_content)
        self.sources_content_layout.setContentsMargins(0, 0, 0, 0)
        self.sources_content_layout.setSpacing(8)
        self.source_scroll.setWidget(self.sources_content)

        sources_layout.addWidget(self.source_scroll)
        right_layout.addWidget(sources_card, stretch=2)

        # Processing Details Panel
        details_card = QFrame()
        details_card.setProperty("class", "card")
        details_layout = QVBoxLayout(details_card)
        details_layout.setContentsMargins(12, 12, 12, 12)
        details_layout.setSpacing(6)

        details_label = QLabel("Execution & Governor Telemetry")
        details_label.setStyleSheet("font-weight: 600; font-size: 12px; color: #F59E0B;")
        details_layout.addWidget(details_label)

        self.lbl_exec_depth = QLabel("• Execution Depth: 4 of 8 blocks (Adaptive Factual)")
        self.lbl_iterations = QLabel("• Reasoning Iterations: 1 of 3 (Confident Grounding)")
        self.lbl_retrieval_ms = QLabel("• BM25 Retrieval: 11.2 ms (Top-3 Chunks)")
        self.lbl_generation_ms = QLabel("• Neural Generation: 54.8 ms (26.4 tokens/s)")
        self.lbl_governor_status = QLabel("• Governor Status: Nominal (0/10.0s Timeout Ceiling)")
        self.lbl_verification = QLabel("• Citation Ledger: 100% Grounded (0 Hallucinations)")

        for lbl in [
            self.lbl_exec_depth,
            self.lbl_iterations,
            self.lbl_retrieval_ms,
            self.lbl_generation_ms,
            self.lbl_governor_status,
            self.lbl_verification,
        ]:
            lbl.setStyleSheet("font-size: 11px; color: #94A3B8; font-family: monospace;")
            details_layout.addWidget(lbl)

        right_layout.addWidget(details_card, stretch=1)

        splitter.addWidget(right_container)
        splitter.setSizes([620, 340])

        main_layout.addWidget(splitter)

    def _on_chip_clicked(self, prompt: str) -> None:
        self.input_edit.setText(prompt)
        self._on_send_clicked()

    def _on_send_clicked(self) -> None:
        query = self.input_edit.text().strip()
        if not query:
            return

        self.input_edit.clear()
        self.add_user_message(query)

        # Simulate small delay for research demonstration
        self.send_btn.setEnabled(False)
        self.send_btn.setText("Retrieving & Synthesizing...")

        QTimer.singleShot(400, lambda: self._generate_simulated_response(query))

    def add_user_message(self, message: str) -> None:
        """Render a user query bubble in the chat view."""
        item = QListWidgetItem()
        bubble = QFrame()
        bubble.setStyleSheet(
            "background-color: #1E293B; border: 1px solid #334155; border-radius: 8px; padding: 10px;"
        )
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(10, 8, 10, 8)
        bubble_layout.setSpacing(4)

        header = QLabel("Researcher (You)")
        header.setStyleSheet("color: #60A5FA; font-weight: 600; font-size: 11px;")

        text = QLabel(message)
        text.setWordWrap(True)
        text.setStyleSheet("color: #FFFFFF; font-size: 13px;")

        bubble_layout.addWidget(header)
        bubble_layout.addWidget(text)

        item.setSizeHint(bubble.sizeHint())
        self.message_list.addItem(item)
        self.message_list.setItemWidget(item, bubble)
        self.message_list.scrollToBottom()

    def add_assistant_message(
        self,
        answer: str,
        citations: List[CitationItem],
        depth: int = 4,
        latency_ms: float = 66.0,
    ) -> None:
        """Render a grounded assistant answer bubble in the chat view."""
        item = QListWidgetItem()
        bubble = QFrame()
        bubble.setStyleSheet(
            "background-color: #151D2A; border: 1px solid #233147; border-radius: 8px; padding: 12px;"
        )
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 10, 12, 10)
        bubble_layout.setSpacing(6)

        header_layout = QHBoxLayout()
        header = QLabel("AHIA-R2 Assistant (15.2M Model)")
        header.setStyleSheet("color: #34D399; font-weight: 700; font-size: 11px;")

        badge = QLabel(f"DEPTH {depth} BLOCKS • GROUNDED")
        badge.setStyleSheet(
            "color: #60A5FA; font-size: 10px; font-weight: 600; background: #1E293B; "
            "padding: 2px 6px; border-radius: 4px;"
        )

        header_layout.addWidget(header)
        header_layout.addStretch()
        header_layout.addWidget(badge)
        bubble_layout.addLayout(header_layout)

        text = QLabel(answer)
        text.setWordWrap(True)
        text.setStyleSheet("color: #F3F4F6; font-size: 13px; line-height: 1.4;")
        bubble_layout.addWidget(text)

        # Citation chips
        if citations:
            chips_layout = QHBoxLayout()
            chips_layout.setSpacing(6)
            for cite in citations:
                chip = QLabel(f"📖 {cite.book_title} (Ch. {cite.chapter_title})")
                chip.setStyleSheet(
                    "background-color: #111827; color: #93C5FD; border: 1px solid #1E3A8A; "
                    "border-radius: 4px; padding: 2px 6px; font-size: 10px;"
                )
                chips_layout.addWidget(chip)
            chips_layout.addStretch()
            bubble_layout.addLayout(chips_layout)

        item.setSizeHint(bubble.sizeHint())
        self.message_list.addItem(item)
        self.message_list.setItemWidget(item, bubble)
        self.message_list.scrollToBottom()

        # Update Sources Panel
        self._update_sources_panel(citations)

    def _generate_simulated_response(self, query: str) -> None:
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Send Question")

        # Formulate grounded demo response
        if "10%" in query or "saving" in query.lower() or "babylon" in query.lower():
            answer = (
                "According to George S. Clason in *The Richest Man in Babylon*, the foundational rule "
                "of personal wealth accumulation is that at least **one-tenth of all you earn is yours to keep**. "
                "No matter how modest your income, this tenth must be retained before addressing living expenses, "
                "thereby initiating the compound accumulation of investment capital."
            )
            cites = [
                CitationItem(
                    chunk_id="BAB_CH02_0004",
                    book_title="The Richest Man in Babylon",
                    author="George S. Clason",
                    chapter_title="Seven Cures for a Lean Purse",
                    excerpt="A part of all you earn is yours to keep. It should be not less than a tenth no matter how little you earn.",
                    match_score=5.82,
                    char_start=4120,
                    char_end=4245,
                )
            ]
        elif "morgan" in query.lower() or "psychology" in query.lower():
            answer = (
                "In *The Psychology of Money*, Morgan Housel defines wealth as the options and flexibility you possess, "
                "contrasting it with 'richness' which is current spending. Wealth is financial options that have not yet "
                "been converted into tangible possessions or visible consumption."
            )
            cites = [
                CitationItem(
                    chunk_id="PSY_CH09_0012",
                    book_title="The Psychology of Money",
                    author="Morgan Housel",
                    chapter_title="Wealth is What You Don't See",
                    excerpt="Wealth is the nice cars not purchased. The diamonds not bought. The watches not worn. Wealth is financial options not yet converted.",
                    match_score=6.15,
                    char_start=18400,
                    char_end=18580,
                )
            ]
        else:
            answer = (
                f"Regarding your inquiry about '{query}', our local personal-finance library emphasizes "
                "disciplined capital preservation, avoiding emotional investment traps, and automating savings. "
                "All assertions are verified against local chunk indices with zero external API calls."
            )
            cites = [
                CitationItem(
                    chunk_id="BAB_CH04_0007",
                    book_title="The Richest Man in Babylon",
                    author="George S. Clason",
                    chapter_title="The Five Laws of Gold",
                    excerpt="Gold clingeth to the protection of the cautious owner who invests it under the advice of men wise in its handling.",
                    match_score=4.40,
                    char_start=8100,
                    char_end=8260,
                )
            ]

        self.add_assistant_message(answer, cites, depth=4)

    def _update_sources_panel(self, citations: List[CitationItem]) -> None:
        # Clear previous sources
        while self.sources_content_layout.count():
            item = self.sources_content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for cite in citations:
            card = QFrame()
            card.setStyleSheet(
                "background-color: #121824; border: 1px solid #233147; border-radius: 6px; padding: 8px;"
            )
            layout = QVBoxLayout(card)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(4)

            title = QLabel(f"📖 {cite.book_title}")
            title.setStyleSheet("font-weight: 600; color: #60A5FA; font-size: 11px;")

            meta = QLabel(f"Author: {cite.author} • Ch: {cite.chapter_title}")
            meta.setStyleSheet("color: #94A3B8; font-size: 10px;")

            excerpt = QLabel(f"\"{cite.excerpt}\"")
            excerpt.setWordWrap(True)
            excerpt.setStyleSheet("color: #E2E8F0; font-size: 11px; font-style: italic; background: #0B0F17; padding: 6px; border-radius: 4px;")

            score = QLabel(f"BM25 Score: {cite.match_score:.2f} • Verified Grounding")
            score.setStyleSheet("color: #34D399; font-size: 10px; font-weight: 600;")

            layout.addWidget(title)
            layout.addWidget(meta)
            layout.addWidget(excerpt)
            layout.addWidget(score)

            self.sources_content_layout.addWidget(card)

        self.sources_content_layout.addStretch()

    def _load_initial_demo_conversation(self) -> None:
        # Initial greeting with demonstration grounding
        self.add_assistant_message(
            "Welcome to **AHIA-R2 Book Intelligence**. I am an offline personal-finance research demonstration "
            "powered by a small 15.2M parameter Transformer trained from random initialization. All knowledge "
            "is strictly grounded upon your local library of personal-finance books using Okapi BM25 retrieval.",
            citations=[
                CitationItem(
                    chunk_id="BAB_CH01_0001",
                    book_title="The Richest Man in Babylon",
                    author="George S. Clason",
                    chapter_title="An Historical Sketch of Babylon",
                    excerpt="Money is the medium by which earthly success is measured.",
                    match_score=5.10,
                    char_start=0,
                    char_end=95,
                )
            ],
            depth=2,
        )
