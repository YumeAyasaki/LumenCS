"""Typed, provider-neutral tools for an LLM or agent to query source materials."""

from __future__ import annotations

import ast
import fnmatch
import logging
import operator
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, DecimalException
from functools import lru_cache
from typing import Annotated, Literal, TypeAlias
from urllib.parse import quote

from fastembed import TextEmbedding
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import Settings
from app.database import (
    Price,
    Product,
    Promotion,
    Region,
    ServiceCentre,
    ServiceFee,
    SourceSheet,
    Warranty,
)
from app.tools.normalized_data import (
    find_service_centres,
    get_prices,
    get_products,
    get_promotions,
    get_promotions_for_model,
    get_regions,
    get_service_fees,
    get_warranty,
)
from app.tools.pdf_retriever import PDFSearchResult, retrieve_policy_chunks
from app.tools.spreadsheet_reader import (
    SchemaMapping,
    WorkbookMapping,
    load_schema_mapping,
)

logger: logging.Logger = logging.getLogger(__name__)


class ToolArguments(BaseModel):
    """Provide hidden server time context and typed argument validation.

    Attributes:
        as_of: Timestamp of the customer's question, used for effective-date
            filtering.

    Raises:
        ValueError: If any declared argument is invalid.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    as_of: datetime = Field(
        description="Thời điểm câu hỏi được đặt, định dạng ISO 8601."
    )


class ProductDetailsArguments(ToolArguments):
    """Describe filters for looking up catalog products.

    Attributes:
        model: Optional exact product model code.
        product_group: Optional exact product group.
    """

    model: str | None = Field(default=None, description="Mã model chính xác.")
    product_group: str | None = Field(
        default=None,
        description="Nhóm sản phẩm chính xác.",
    )


class QuotePriceArguments(ToolArguments):
    """Describe the product and location for a price lookup.

    Attributes:
        model: Exact product model code.
        province: Optional province or city used to resolve the price region.
        region: Optional exact price region.
        channel: Optional sales channel for promotion eligibility.
    """

    model: str = Field(min_length=1, description="Mã model cần báo giá.")
    province: str | None = Field(
        default=None,
        description="Tỉnh/thành để tự tra cứu vùng giá.",
    )
    region: str | None = Field(
        default=None,
        description="Vùng giá chính xác nếu đã biết.",
    )
    channel: str | None = Field(
        default=None,
        description="Kênh mua hàng (ví dụ Online hoặc tại cửa hàng).",
    )


class PromotionArguments(ToolArguments):
    """Describe filters for active promotion lookup.

    Attributes:
        model: Optional product model used to match promotion scope.
        channel: Optional sales channel.
    """

    model: str | None = Field(default=None, description="Mã model sản phẩm.")
    channel: str | None = Field(default=None, description="Kênh bán hàng.")


class WarrantyArguments(ToolArguments):
    """Describe filters for warranty lookup.

    Attributes:
        model: Optional model code to resolve to its product group.
        product_group: Optional exact product group.
    """

    model: str | None = Field(default=None, description="Mã model sản phẩm.")
    product_group: str | None = Field(
        default=None,
        description="Nhóm sản phẩm chính xác.",
    )


class ServiceCentreArguments(ToolArguments):
    """Describe filters for service-centre lookup.

    Attributes:
        province: Optional exact province or city.
        district: Optional district within the province.
        product_group: Optional product group the centre must support.
    """

    province: str | None = Field(default=None, description="Tỉnh/thành.")
    district: str | None = Field(default=None, description="Quận/huyện.")
    product_group: str | None = Field(
        default=None,
        description="Nhóm sản phẩm cần được trung tâm tiếp nhận.",
    )


class ServiceFeeArguments(ToolArguments):
    """Describe filters for service-fee lookup.

    Attributes:
        service_code: Optional exact fee code.
        service_query: Optional text contained in a service name.
        product_group: Optional product group to check for applicability.
    """

    service_code: str | None = Field(default=None, description="Mã dịch vụ.")
    service_query: str | None = Field(
        default=None,
        min_length=1,
        description="Một phần tên dịch vụ.",
    )
    product_group: str | None = Field(
        default=None,
        description="Nhóm sản phẩm cần kiểm tra phạm vi áp dụng.",
    )


class PolicySearchArguments(ToolArguments):
    """Describe a semantic search in the policy handbook.

    Attributes:
        query: Vietnamese question or phrase to search for.
    """

    query: str = Field(
        min_length=1,
        description="Câu hỏi cần tra trong sổ tay.",
    )


class CalculateArguments(ToolArguments):
    """Describe a safe arithmetic expression to evaluate.

    Attributes:
        expression: Arithmetic using numeric literals and +, -, *, /, and
            parentheses.
    """

    expression: str = Field(
        min_length=1,
        max_length=256,
        description="Biểu thức số học; chỉ hỗ trợ +, -, *, / và dấu ngoặc.",
    )


class AddDaysArguments(ToolArguments):
    """Request a date offset calculation.

    Attributes:
        start_date: Start date in ISO YYYY-MM-DD format.
        days: Number of calendar days to add; negative values subtract days.
    """

    start_date: date = Field(description="Ngày bắt đầu, định dạng YYYY-MM-DD.")
    days: int = Field(description="Số ngày cần cộng hoặc trừ.")


class CitedToolRecord(BaseModel):
    """Represent one record with source provenance.

    Attributes:
        kind: Tool-result record type.
        cite_id: Stable source citation for this record.

    Raises:
        ValidationError: If a citation is missing or malformed.
    """

    cite_id: str


class ProductRecord(CitedToolRecord):
    """Represent a product catalog record returned to the model.

    Attributes:
        kind: Product record discriminator.
        model: Catalog model code.
        name: Product display name.
        product_group: Normalized product group.
        specification: Published specification, if available.
        energy_rating: Published energy rating, if available.
        dimensions: Published dimensions, if available.
        weight: Published weight, if available.
        status: Catalog status, if available.
        notes: Source notes, if available.
        cite_id: Workbook, sheet, and source row citation.

    Raises:
        ValidationError: If returned values do not match the record schema.
    """

    kind: Literal["product"] = "product"
    model: str
    name: str
    product_group: str
    specification: str | None
    energy_rating: str | None
    dimensions: str | None
    weight: str | None
    status: str | None
    notes: str | None


class PriceRecord(CitedToolRecord):
    """Represent an effective product price returned to the model.

    Attributes:
        kind: Price record discriminator.
        model: Catalog model code.
        region: Price region, if a price was found.
        price: Effective listed price, if a price was found.
        valid_from: Date the listed price became effective.
        product_status: Catalog status of the model.
        product_notes: Catalog note, including replacement model details.
        product_cite_id: Product catalog citation.
        promotions: Applicable or conditional price reductions.
        not_applicable: Programs excluded by date, model, or channel.
        installment_note: Eligibility terms for applicable installments.
        installment_cite_ids: Promotion citations supporting installment terms.
        calculation_note: Explains that each discount is calculated alone.
        cite_id: Workbook, sheet, and source row citation.

    Raises:
        ValidationError: If returned values do not match the record schema.
    """

    kind: Literal["price"] = "price"
    model: str
    region: str | None
    price: Decimal | None
    valid_from: date | None
    product_status: str | None = None
    product_notes: str | None = None
    product_cite_id: str | None = None
    promotions: list[PricePromotionRecord] = Field(default_factory=list)
    not_applicable: list[NotApplicablePromotionRecord] = Field(
        default_factory=list
    )
    installment_note: str | None = None
    installment_cite_ids: list[str] = Field(default_factory=list)
    calculation_note: str | None = None


class PricePromotionRecord(BaseModel):
    """Describe one price discount and its independently calculated impact.

    Attributes:
        code: Source promotion code.
        name: Promotion name.
        channel: Source channel limitation, if any.
        discount_amount: Discount after applying any source cap, if price exists.
        price_after_discount: Listed price less this promotion only, if available.
        conditions: Eligibility terms from the source.
        stacking_text: Source stacking rule.
        channel_conditional: Whether the customer channel is unknown.
        cite_id: Promotion source citation.
    """

    code: str
    name: str
    channel: str | None
    discount_amount: Decimal | None
    price_after_discount: Decimal | None
    conditions: str | None
    stacking_text: str | None
    channel_conditional: bool
    cite_id: str


class NotApplicablePromotionRecord(BaseModel):
    """Explain why a promotion matching a model cannot be used.

    Attributes:
        code: Source promotion code.
        name: Promotion name.
        reason: Date, model-exclusion, or channel reason.
        cite_id: Promotion source citation.
    """

    code: str
    name: str
    reason: str
    cite_id: str


class PromotionRecord(CitedToolRecord):
    """Represent a promotion record and parsed scope.

    Attributes:
        kind: Promotion record discriminator.
        code: Source promotion code.
        name: Promotion name.
        starts_on: First date the promotion is active.
        ends_on: Last date the promotion is active.
        scope_text: Original source applicability text.
        scope: Parsed included and excluded model patterns.
        channel: Applicable sales channel, if specified.
        promotion_type: Parsed promotion category.
        value_amount: Parsed numeric value, if available.
        value_unit: Unit of the promotion value, if available.
        value_details: Original value details.
        cap_amount: Parsed maximum discount, if available.
        conditions: Source conditions, if available.
        can_stack: Whether the promotion can be combined.
        stacking_text: Original stacking text, if available.
        cite_id: Workbook, sheet, and source row citation.

    Raises:
        ValidationError: If returned values do not match the record schema.
    """

    kind: Literal["promotion"] = "promotion"
    code: str
    name: str
    starts_on: date
    ends_on: date
    scope_text: str
    scope: dict[str, list[str]]
    channel: str | None
    promotion_type: str
    value_amount: Decimal | None
    value_unit: str | None
    value_details: str | None
    cap_amount: Decimal | None
    conditions: str | None
    can_stack: bool
    stacking_text: str | None


class WarrantyRecord(CitedToolRecord):
    """Represent a warranty duration and its conditions.

    Attributes:
        kind: Warranty record discriminator.
        product_group: Applicable product group.
        product_months: Product warranty duration in months.
        component: Covered component, if specified.
        component_months: Component warranty duration, if specified.
        exclusions: Source exclusions, if specified.
        cite_id: Workbook, sheet, and source row citation.

    Raises:
        ValidationError: If returned values do not match the record schema.
    """

    kind: Literal["warranty"] = "warranty"
    product_group: str
    product_months: int
    component: str | None
    component_months: int | None
    exclusions: str | None


class ServiceCentreRecord(CitedToolRecord):
    """Represent an eligible service centre and its contact details.

    Attributes:
        kind: Service-centre record discriminator.
        code: Source centre code.
        name: Centre name.
        province: Province or city.
        district: District, if available.
        address: Source address.
        phone: Source phone number, if available.
        weekday_hours: Weekday hours, if available.
        sunday_hours: Sunday hours, if available.
        accepted_services: Product groups accepted by the centre.
        status: Source status, if available.
        notes: Source notes, if available.
        cite_id: Workbook, sheet, and source row citation.

    Raises:
        ValidationError: If returned values do not match the record schema.
    """

    kind: Literal["service_centre"] = "service_centre"
    code: str
    name: str
    province: str
    district: str | None
    address: str
    phone: str | None
    weekday_hours: str | None
    sunday_hours: str | None
    accepted_services: dict[str, bool]
    status: str | None
    notes: str | None


class ServiceFeeRecord(CitedToolRecord):
    """Represent in-warranty and out-of-warranty service fees.

    Attributes:
        kind: Service-fee record discriminator.
        code: Service fee code.
        service: Service name.
        applies_to: Applicable product groups.
        unit: Fee unit, if specified.
        fee_in_warranty: In-warranty fee, if specified.
        fee_out_of_warranty: Out-of-warranty fee, if specified.
        fee_in_warranty_note: Source note for the in-warranty fee.
        fee_out_of_warranty_note: Source note for the out-of-warranty fee.
        notes: Other source notes.
        cite_id: Workbook, sheet, and source row citation.

    Raises:
        ValidationError: If returned values do not match the record schema.
    """

    kind: Literal["service_fee"] = "service_fee"
    code: str
    service: str
    applies_to: list[str]
    unit: str | None
    fee_in_warranty: Decimal | None
    fee_out_of_warranty: Decimal | None
    fee_in_warranty_note: str | None
    fee_out_of_warranty_note: str | None
    notes: str | None


class PolicyRecord(CitedToolRecord):
    """Represent a handbook excerpt and its clause citation.

    Attributes:
        kind: Policy record discriminator.
        source_file: Relative PDF path.
        section_id: Clause or section anchor.
        parent_section_id: Top-level section containing the matched clause.
        section_title: Handbook section title.
        page: One-based PDF page number.
        text: Complete section containing the matched clause.
        section_cite_ids: Clause citations included in the expanded section.
        document_version: Handbook version, if available.
        effective_date: Handbook effective date, if available.
        cite_id: Stable policy clause citation.

    Raises:
        ValidationError: If returned values do not match the record schema.
    """

    kind: Literal["policy"] = "policy"
    source_file: str
    section_id: str
    parent_section_id: str
    section_title: str
    page: int
    text: str
    section_cite_ids: list[str]
    document_version: str | None
    effective_date: str | None


class CalculationRecord(BaseModel):
    """Represent a calculated numeric result.

    Attributes:
        kind: Identifies this record as a calculation.
        expression: The evaluated expression.
        result: Exact decimal result.
    """

    kind: Literal["calculation"] = "calculation"
    expression: str
    result: Decimal


class AddDaysRecord(BaseModel):
    """Represent a date offset and its Vietnamese weekday.

    Attributes:
        kind: Identifies this record as a date calculation.
        start_date: Supplied starting date.
        days: Number of days added.
        result_date: Resulting calendar date.
        weekday: Vietnamese weekday name for the result.
    """

    kind: Literal["add_days"] = "add_days"
    start_date: date
    days: int
    result_date: date
    weekday: str


ToolRecord: TypeAlias = Annotated[
    ProductRecord
    | PriceRecord
    | PromotionRecord
    | WarrantyRecord
    | ServiceCentreRecord
    | ServiceFeeRecord
    | PolicyRecord
    | CalculationRecord
    | AddDaysRecord,
    Field(discriminator="kind"),
]


class ToolResult(BaseModel):
    """Return typed records and the exact citations available to the model.

    Attributes:
        tool_name: Name of the tool that was executed.
        records: Typed results from the corresponding source.
        citations: Unique citations attached to returned source records.
        notes: Compact worksheet notes in ``[cite_id] text`` form.
    """

    tool_name: str
    records: list[ToolRecord]
    citations: list[str]
    notes: list[str] = Field(default_factory=list)


class FunctionDefinition(BaseModel):
    """Describe one function tool in a provider-neutral function-call shape.

    Attributes:
        name: Callable tool name.
        description: Guidance shown to the model.
        parameters: JSON Schema accepted by the tool.

    Raises:
        ValidationError: If the function declaration is malformed.
    """

    name: str
    description: str
    parameters: dict[str, object]


class ModelToolDefinition(BaseModel):
    """Represent a model-callable function declaration.

    Attributes:
        type: Function tool discriminator.
        function: Name, description, and JSON Schema for invocation.

    Raises:
        ValidationError: If the function declaration is malformed.
    """

    type: Literal["function"] = "function"
    function: FunctionDefinition


class ToolInvocationRequest(BaseModel):
    """Validate a tool invocation payload received by the HTTP API.

    Attributes:
        arguments: JSON arguments validated against the named tool's schema.
    """

    arguments: dict[str, object]


class ToolInvocationCall(BaseModel):
    """Describe the tool call visible to a client.

    Attributes:
        name: Registered tool name that was invoked.
        arguments: Arguments used for the call, including the effective date
            when the model supplied a valid JSON object.
    """

    name: str
    arguments: dict[str, object] | str


class ToolInvocationResponse(BaseModel):
    """Return both the executed tool call and its structured answer.

    Attributes:
        call: Tool name and arguments used for this invocation.
        answer: Typed source records and citations returned by the tool.
    """

    call: ToolInvocationCall
    answer: ToolResult


@dataclass(frozen=True)
class ModelToolContext:
    """Hold the clients required by tool handlers for one invocation.

    Attributes:
        session: Active asynchronous PostgreSQL session.
        qdrant_client: Connected asynchronous Qdrant client.
        embedding_model: Loaded multilingual embedding model.
        settings: Application settings for policy retrieval.
    """

    session: AsyncSession
    qdrant_client: AsyncQdrantClient
    embedding_model: TextEmbedding
    settings: Settings


TOOL_DESCRIPTIONS: dict[str, tuple[type[ToolArguments], str]] = {
    "get_product_details": (
        ProductDetailsArguments,
        (
            "Dùng để xác minh tên, nhóm, thông số hoặc trạng thái một model; trả "
            "về các trường trong danh mục kèm nguồn. Không gọi thêm nếu quote_price "
            "đã trả trạng thái model."
        ),
    ),
    "quote_price": (
        QuotePriceArguments,
        (
            "Dùng để báo giá model theo tỉnh/thành hoặc vùng; trả giá niêm yết, "
            "khuyến mãi phù hợp, mức giảm đã tính và giá sau từng mức giảm. Không "
            "gọi thêm công cụ khuyến mãi cho cùng yêu cầu giá."
        ),
    ),
    "get_promotions": (
        PromotionArguments,
        (
            "Dùng khi khách hỏi chương trình khuyến mãi nói chung hoặc điều kiện "
            "không cần báo giá; trả chương trình đang hiệu lực theo ngày/model/kênh. "
            "Nếu đang báo giá model, dùng quote_price thay vì gọi thêm công cụ này."
        ),
    ),
    "get_warranty": (
        WarrantyArguments,
        (
            "Dùng để tra thời hạn, linh kiện và loại trừ bảo hành theo model/nhóm; "
            "trả điều khoản có trích dẫn. Không gọi get_product_details chỉ để tìm "
            "nhóm của một model."
        ),
    ),
    "find_service_centres": (
        ServiceCentreArguments,
        (
            "Dùng để tìm địa chỉ, điện thoại, giờ làm và năng lực trung tâm theo "
            "tỉnh/quận/nhóm; kết quả chỉ gồm trung tâm khớp bộ lọc."
        ),
    ),
    "get_service_fee": (
        ServiceFeeArguments,
        (
            "Dùng để tra phí dịch vụ trong/ngoài bảo hành theo mã, tên dịch vụ "
            "hoặc nhóm sản phẩm; không suy ra phí từ trung tâm dịch vụ."
        ),
    ),
    "search_policy": (
        PolicySearchArguments,
        (
            "Dùng để tìm chính sách, quy trình, đổi trả, lắp đặt hoặc hỗ trợ trong "
            "sổ tay; trả toàn bộ mục cha có liên quan và cite_id. Không tự suy diễn "
            "điều khoản ngoài nội dung trả về."
        ),
    ),
    "calculate": (
        CalculateArguments,
        (
            "Dùng để tính số học từ giá trị đã được nguồn/công cụ cung cấp; trả "
            "kết quả Decimal chính xác. Không tự tính giảm giá hoặc tổng tiền."
        ),
    ),
    "add_days": (
        AddDaysArguments,
        (
            "Dùng khi cần cộng/trừ số ngày lịch từ một ngày đã biết; trả ngày kết "
            "quả và thứ trong tuần. Ngày hỏi hiện tại đã có trong hướng dẫn hệ thống."
        ),
    ),
}


class UnknownModelToolError(ValueError):
    """Indicate that a requested model tool is not registered."""


def get_model_tool_definitions() -> list[ModelToolDefinition]:
    """Return model-facing JSON-schema function declarations.

    Returns:
        list[ModelToolDefinition]: Tool names, descriptions, and input schemas.
    """
    def clean_schema(value: object) -> object:
        """Remove schema titles and the server-injected timestamp recursively."""
        if isinstance(value, dict):
            return {
                key: clean_schema(item)
                for key, item in value.items()
                if key != "title"
            }
        if isinstance(value, list):
            return [clean_schema(item) for item in value]
        return value

    definitions: list[ModelToolDefinition] = []
    for name, (arguments_model, description) in TOOL_DESCRIPTIONS.items():
        schema: dict[str, object] = arguments_model.model_json_schema()
        properties: object = schema.get("properties")
        if isinstance(properties, dict):
            properties.pop("as_of", None)
        required: object = schema.get("required")
        if isinstance(required, list):
            required_fields: list[object] = [
                key for key in required if key != "as_of"
            ]
            if required_fields:
                schema["required"] = required_fields
            else:
                schema.pop("required")
        cleaned_schema: object = clean_schema(schema)
        if not isinstance(cleaned_schema, dict):
            raise TypeError("Tool argument schema must be a JSON object.")
        definitions.append(
            ModelToolDefinition(
                function=FunctionDefinition(
                    name=name,
                    description=description,
                    parameters=cleaned_schema,
                )
            )
        )
    return definitions


def _record_from_product(product: Product) -> ProductRecord:
    """Convert a typed product row into a model-facing result record.

    Args:
        product: Product row with eagerly-loaded source provenance.

    Returns:
        ProductRecord: Product details and citation.
    """
    return ProductRecord(
        model=product.model,
        name=product.name,
        product_group=product.product_group,
        specification=product.specification,
        energy_rating=product.energy_rating,
        dimensions=product.dimensions,
        weight=product.weight,
        status=product.status,
        notes=product.notes,
        cite_id=product.cite_id,
    )


def _promotion_applies_to_model(promotion: Promotion, model: str) -> bool:
    """Return whether a promotion's include patterns contain a model.

    Args:
        promotion: Promotion with parsed include/exclude patterns.
        model: Exact product model code.

    Returns:
        ``True`` when included and not explicitly excluded.
    """
    included: bool = any(
        fnmatch.fnmatchcase(model, pattern)
        for pattern in promotion.scope.get("include", [])
    )
    excluded: bool = any(
        fnmatch.fnmatchcase(model, pattern)
        for pattern in promotion.scope.get("exclude", [])
    )
    return included and not excluded


def _discount_for_price(
    promotion: Promotion,
    price: Decimal,
) -> Decimal | None:
    """Calculate an individual cash or percentage discount with its cap.

    Args:
        promotion: Promotion with normalized amount, unit, and cap.
        price: Listed price before this promotion.

    Returns:
        Discount rounded to whole currency units, or ``None`` for programs
        that do not directly reduce price.
    """
    if promotion.promotion_type == "percentage_discount":
        if promotion.value_amount is None:
            return None
        discount: Decimal = price * promotion.value_amount / Decimal(100)
        if promotion.cap_amount is not None:
            discount = min(discount, promotion.cap_amount)
    elif promotion.promotion_type == "cash_discount":
        if promotion.value_amount is None:
            return None
        discount = min(price, promotion.value_amount)
    else:
        return None
    return discount.quantize(Decimal(1), rounding=ROUND_HALF_UP)


def _quote_promotions(
    promotions: list[Promotion],
    model: str,
    price: Decimal | None,
    as_of: datetime,
    channel: str | None,
) -> tuple[
    list[PricePromotionRecord],
    list[NotApplicablePromotionRecord],
    str | None,
    list[str],
]:
    """Classify model promotions and calculate independent price reductions.

    Args:
        promotions: All dated promotion rows whose patterns mention the model.
        model: Exact model code being quoted.
        price: Listed price, if one was found.
        as_of: Authoritative timestamp for the question.
        channel: Customer's sales channel, if known.

    Returns:
        Applicable/conditional price reductions, inapplicable programs, and
        installment eligibility terms.
    """
    applicable: list[PricePromotionRecord] = []
    not_applicable: list[NotApplicablePromotionRecord] = []
    installment_conditions: list[str] = []
    installment_citations: list[str] = []
    asked_date: date = as_of.date()
    for promotion in promotions:
        excluded_patterns: list[str] = [
            pattern
            for pattern in promotion.scope.get("exclude", [])
            if fnmatch.fnmatchcase(model, pattern)
        ]
        if excluded_patterns:
            reason: str = f"excluded: {', '.join(excluded_patterns)}"
        elif not _promotion_applies_to_model(promotion, model):
            continue
        elif promotion.ends_on < asked_date:
            reason = f"ended {promotion.ends_on.strftime('%d/%m/%Y')}"
        elif promotion.starts_on > asked_date:
            reason = f"starts {promotion.starts_on.strftime('%d/%m/%Y')}"
        elif (
            channel is not None
            and promotion.channel not in (None, "Tất cả")
            and promotion.channel.casefold() != channel.casefold()
        ):
            reason = f"channel limited to {promotion.channel}"
        else:
            if promotion.promotion_type == "installment":
                condition: str = (
                    promotion.conditions
                    or promotion.value_details
                    or promotion.name
                )
                if channel is None and promotion.channel not in (None, "Tất cả"):
                    condition = f"{condition} (kênh {promotion.channel}, cần xác nhận)"
                installment_conditions.append(condition)
                installment_citations.append(promotion.cite_id)
                continue
            if promotion.promotion_type not in {
                "percentage_discount",
                "cash_discount",
            }:
                continue
            discount: Decimal | None = (
                _discount_for_price(promotion, price)
                if price is not None
                else None
            )
            if price is not None and discount is None:
                continue
            applicable.append(
                PricePromotionRecord(
                    code=promotion.code,
                    name=promotion.name,
                    channel=promotion.channel,
                    discount_amount=discount,
                    price_after_discount=(
                        price - discount
                        if price is not None and discount is not None
                        else None
                    ),
                    conditions=promotion.conditions,
                    stacking_text=promotion.stacking_text,
                    channel_conditional=(
                        channel is None
                        and promotion.channel not in (None, "Tất cả")
                    ),
                    cite_id=promotion.cite_id,
                )
            )
            continue
        not_applicable.append(
            NotApplicablePromotionRecord(
                code=promotion.code,
                name=promotion.name,
                reason=reason,
                cite_id=promotion.cite_id,
            )
        )
    return (
        applicable,
        not_applicable,
        "; ".join(dict.fromkeys(installment_conditions)) or None,
        list(dict.fromkeys(installment_citations)),
    )


def _quote_price_record(
    model: str,
    price: Price | None,
    product: Product | None,
    region: str | None,
    promotions: list[Promotion],
    as_of: datetime,
    channel: str | None,
) -> PriceRecord:
    """Build a quote record containing price, promotions, and catalog status.

    Args:
        model: Exact product model code.
        price: Effective listed price, if available.
        product: Catalog record, if the model exists.
        region: Resolved price region, if available.
        promotions: Programs whose include/exclude patterns mention the model.
        as_of: Authoritative timestamp for the question.
        channel: Customer sales channel, if known.

    Returns:
        Typed quote record, including explanations for price/promotion gaps.
    """
    listed_price: Decimal | None = price.price if price is not None else None
    if price is None and product is None:
        raise ValueError("A quote record requires a price or catalog product.")
    (
        applicable,
        not_applicable,
        installment_note,
        installment_cite_ids,
    ) = _quote_promotions(
        promotions,
        model,
        listed_price,
        as_of,
        channel,
    )
    product_notes: str | None = product.notes if product is not None else None
    cite_id: str = (
        price.cite_id if price is not None else product.cite_id
    )
    return PriceRecord(
        cite_id=cite_id,
        model=model,
        region=region,
        price=listed_price,
        valid_from=price.valid_from if price is not None else None,
        product_status=product.status if product is not None else None,
        product_notes=product_notes,
        product_cite_id=product.cite_id if product is not None else None,
        promotions=applicable,
        not_applicable=not_applicable,
        installment_note=installment_note,
        installment_cite_ids=installment_cite_ids,
        calculation_note=(
            "Mỗi khuyến mãi được tính riêng trên giá niêm yết; chưa cộng dồn."
            if applicable
            else None
        ),
    )


def _record_from_promotion(promotion: Promotion) -> PromotionRecord:
    """Convert a promotion row into a model-facing result record.

    Args:
        promotion: Active promotion with eagerly-loaded source provenance.

    Returns:
        PromotionRecord: Promotion terms and citation.
    """
    return PromotionRecord(
        code=promotion.code,
        name=promotion.name,
        starts_on=promotion.starts_on,
        ends_on=promotion.ends_on,
        scope_text=promotion.scope_text,
        scope=promotion.scope,
        channel=promotion.channel,
        promotion_type=promotion.promotion_type,
        value_amount=promotion.value_amount,
        value_unit=promotion.value_unit,
        value_details=promotion.value_details,
        cap_amount=promotion.cap_amount,
        conditions=promotion.conditions,
        can_stack=promotion.can_stack,
        stacking_text=promotion.stacking_text,
        cite_id=promotion.cite_id,
    )


def _record_from_warranty(warranty: Warranty) -> WarrantyRecord:
    """Convert a warranty row into a model-facing result record.

    Args:
        warranty: Warranty row with eagerly-loaded source provenance.

    Returns:
        WarrantyRecord: Warranty durations and citation.
    """
    return WarrantyRecord(
        product_group=warranty.product_group,
        product_months=warranty.product_months,
        component=warranty.component,
        component_months=warranty.component_months,
        exclusions=warranty.exclusions,
        cite_id=warranty.cite_id,
    )


def _record_from_centre(centre: ServiceCentre) -> ServiceCentreRecord:
    """Convert a service centre row into a model-facing result record.

    Args:
        centre: Centre row with eagerly-loaded source provenance.

    Returns:
        ServiceCentreRecord: Centre details and citation.
    """
    return ServiceCentreRecord(
        code=centre.code,
        name=centre.name,
        province=centre.province,
        district=centre.district,
        address=centre.address,
        phone=centre.phone,
        weekday_hours=centre.weekday_hours,
        sunday_hours=centre.sunday_hours,
        accepted_services=centre.accepted_services,
        status=centre.status,
        notes=centre.notes,
        cite_id=centre.cite_id,
    )


def _record_from_fee(fee: ServiceFee) -> ServiceFeeRecord:
    """Convert a service-fee row into a model-facing result record.

    Args:
        fee: Fee row with eagerly-loaded source provenance.

    Returns:
        ServiceFeeRecord: Fee details and citation.
    """
    return ServiceFeeRecord(
        code=fee.code,
        service=fee.service,
        applies_to=fee.applies_to,
        unit=fee.unit,
        fee_in_warranty=fee.fee_in_warranty,
        fee_out_of_warranty=fee.fee_out_of_warranty,
        fee_in_warranty_note=fee.fee_in_warranty_note,
        fee_out_of_warranty_note=fee.fee_out_of_warranty_note,
        notes=fee.notes,
        cite_id=fee.cite_id,
    )


def _calculate_expression(expression: str) -> Decimal:
    """Evaluate a small arithmetic expression without executing arbitrary code.

    Args:
        expression: Expression containing numeric literals and +, -, *, /.

    Returns:
        Decimal: Decimal result within the configured Decimal context.

    Raises:
        ValueError: If the expression contains unsupported syntax or divides
            by zero.
    """
    try:
        tree: ast.Expression = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("Invalid arithmetic expression.") from error

    binary_operations: dict[
        type[ast.operator],
        Callable[[Decimal, Decimal], Decimal],
    ] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    unary_operations: dict[type[ast.unaryop], Callable[[Decimal], Decimal]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def evaluate(node: ast.expr) -> Decimal:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal: str | None = ast.get_source_segment(expression, node)
            if literal is None:
                raise ValueError("Invalid numeric literal.")
            return Decimal(literal)
        if isinstance(node, ast.BinOp) and type(node.op) in binary_operations:
            left: Decimal = evaluate(node.left)
            right: Decimal = evaluate(node.right)
            if isinstance(node.op, ast.Div) and right == 0:
                raise ValueError("Division by zero is not allowed.")
            operation: Callable[[Decimal, Decimal], Decimal] = binary_operations[
                type(node.op)
            ]
            return operation(left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_operations:
            operation: Callable[[Decimal], Decimal] = unary_operations[
                type(node.op)
            ]
            return operation(evaluate(node.operand))
        raise ValueError("Expression contains unsupported arithmetic syntax.")

    try:
        result: Decimal = evaluate(tree.body)
    except DecimalException as error:
        raise ValueError(
            "Arithmetic result is outside the supported range."
        ) from error
    if not result.is_finite() or (result != 0 and abs(result.adjusted()) > 1000):
        raise ValueError("Arithmetic result is outside the supported range.")
    return result


SourceEntity: TypeAlias = (
    Product | Price | Promotion | Region | ServiceCentre | ServiceFee | Warranty
)


TOOL_SOURCE_TABLES: dict[str, set[str]] = {
    "get_product_details": {"products"},
    "quote_price": {"products", "prices", "regions", "promotions"},
    "get_promotions": {"promotions"},
    "get_warranty": {"products", "warranty"},
    "find_service_centres": {"service_centres"},
    "get_service_fee": {"service_fees"},
}


@lru_cache(maxsize=1)
def _schema_mapping() -> SchemaMapping:
    """Load the reviewed spreadsheet mapping once for tool-note lookup."""
    return load_schema_mapping()


def _sheet_notes(sheet: SourceSheet) -> list[tuple[str, str]]:
    """Format non-title header and footer notes with row citations.

    Args:
        sheet: Persisted worksheet whose queried table is relevant.

    Returns:
        Note citation and compact ``[cite_id] text`` pairs.

    Raises:
        TypeError: If persisted note rows have an invalid shape.
    """
    encoded_sheet: str = quote(sheet.name, safe="")
    source_file: str = sheet.workbook.source_file
    result: list[tuple[str, str]] = []
    for note in [*(sheet.header_notes or []), *(sheet.footnotes or [])]:
        source_row: object = note.get("source_row")
        values: object = note.get("values")
        if not isinstance(source_row, int) or not isinstance(values, dict):
            raise TypeError(
                f"Invalid persisted worksheet note for {source_file} "
                f"sheet {sheet.name!r}."
            )
        text: str = " ".join(
            str(value).strip()
            for value in values.values()
            if value is not None and str(value).strip()
        )
        if not text:
            continue
        if len(values) == 1 and text.isupper() and any(
            character.isalpha() for character in text
        ):
            continue
        cite_id: str = (
            f"{source_file}#sheet={encoded_sheet}&row={source_row}"
        )
        result.append((cite_id, f"[{cite_id}] {text}"))
    return result


def _matching_sheet_tables(source_file: str) -> dict[str, str]:
    """Map worksheet names to normalized tables for a workbook filename.

    Args:
        source_file: Ingested workbook path.

    Returns:
        Configured sheet name to table name pairs.

    Raises:
        ValueError: If the workbook filename matches multiple schema entries.
    """
    matches: list[WorkbookMapping] = [
        workbook_mapping
        for pattern, workbook_mapping in _schema_mapping().files.items()
        if fnmatch.fnmatchcase(source_file.rsplit("/", 1)[-1], pattern)
    ]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple schema mappings match workbook {source_file!r}."
        )
    if not matches:
        return {}
    workbook_mapping = matches[0]
    return {
        sheet_name: sheet_mapping.normalization.table
        for sheet_name, sheet_mapping in workbook_mapping.sheets.items()
        if not sheet_mapping.archive and sheet_mapping.normalization is not None
    }


async def _queried_worksheet_notes(
    session: AsyncSession,
    tool_name: str,
) -> list[tuple[str, str]]:
    """Fetch notes from every configured sheet queried by a tool.

    Args:
        session: Active asynchronous database session.
        tool_name: Registered tool whose normalized source tables were queried.

    Returns:
        Compact note citation and text pairs, including for empty result sets.

    Raises:
        ValueError: If source note data or workbook mapping is invalid.
        SQLAlchemyError: If PostgreSQL cannot execute the query.
    """
    tables: set[str] = TOOL_SOURCE_TABLES.get(tool_name, set())
    if not tables:
        return []
    result = await session.scalars(
        select(SourceSheet)
        .join(SourceSheet.workbook)
        .options(selectinload(SourceSheet.workbook))
        .order_by(SourceSheet.workbook_id, SourceSheet.ordinal)
    )
    notes: list[tuple[str, str]] = []
    for sheet in result.all():
        mapped_tables: dict[str, str] = _matching_sheet_tables(
            sheet.workbook.source_file
        )
        if mapped_tables.get(sheet.name) in tables:
            notes.extend(_sheet_notes(sheet))
    return notes


def _entity_notes(source_entities: list[SourceEntity]) -> list[tuple[str, str]]:
    """Collect note text from already-loaded row provenance.

    Args:
        source_entities: Normalized source rows with eager-loaded provenance.

    Returns:
        Unique compact note citation and text pairs.
    """
    sheets: dict[int, SourceSheet] = {}
    for entity in source_entities:
        sheet: SourceSheet = entity.source_row.sheet
        sheets.setdefault(sheet.id, sheet)
    return [
        note
        for sheet in sheets.values()
        for note in _sheet_notes(sheet)
    ]


def _make_result(
    name: str,
    records: list[ToolRecord],
    source_entities: list[SourceEntity] | None = None,
    source_notes: list[tuple[str, str]] | None = None,
) -> ToolResult:
    """Build a tool response with row citations and compact sheet notes.

    Args:
        name: Registered tool name.
        records: Typed result records.
        source_entities: Optional normalized rows used to resolve worksheets.
        source_notes: Worksheet notes discovered from configured queried sheets.

    Returns:
        ToolResult: Records and citations limited to this invocation.
    """
    citation_values: list[str] = []
    for record in records:
        if isinstance(record, PolicyRecord):
            citation_values.extend(record.section_cite_ids)
        elif isinstance(record, CitedToolRecord):
            citation_values.append(record.cite_id)
        if isinstance(record, PriceRecord):
            if record.product_cite_id is not None:
                citation_values.append(record.product_cite_id)
            citation_values.extend(
                promotion.cite_id for promotion in record.promotions
            )
            citation_values.extend(
                promotion.cite_id for promotion in record.not_applicable
            )
            citation_values.extend(record.installment_cite_ids)
    note_pairs: list[tuple[str, str]] = [
        *(source_notes or []),
        *_entity_notes(source_entities or []),
    ]
    unique_notes: dict[str, str] = {}
    for cite_id, text in note_pairs:
        unique_notes.setdefault(cite_id, text)
        citation_values.append(cite_id)
    citations: list[str] = list(dict.fromkeys(citation_values))
    return ToolResult(
        tool_name=name,
        records=records,
        citations=citations,
        notes=list(unique_notes.values()),
    )


async def execute_model_tool(
    name: str,
    arguments: dict[str, object],
    context: ModelToolContext,
) -> ToolResult:
    """Validate and execute one registered tool using typed source queries.

    Args:
        name: Tool name returned by the model.
        arguments: JSON arguments from the model function call.
        context: Connected database, vector-store, and embedding clients.

    Returns:
        ToolResult: Typed data records and citations for the current call only.

    Raises:
        UnknownModelToolError: If ``name`` is not a registered tool.
        pydantic.ValidationError: If arguments do not match that tool's schema.
        ValueError: If a calculation expression is invalid.
        SQLAlchemyError: If the typed PostgreSQL query fails.
        ApiException: If policy retrieval fails.
    """
    registration: tuple[type[ToolArguments], str] | None = TOOL_DESCRIPTIONS.get(
        name
    )
    if registration is None:
        raise UnknownModelToolError(f"Unknown model tool: {name}")
    arguments_model: type[ToolArguments] = registration[0]
    unknown_arguments: set[str] = set(arguments).difference(
        arguments_model.model_fields
    )
    if unknown_arguments:
        logger.info(
            "Ignoring unknown arguments for tool %s: %s",
            name,
            sorted(unknown_arguments),
        )
    parsed: ToolArguments = arguments_model.model_validate(arguments)
    as_of: datetime = parsed.as_of
    session: AsyncSession = context.session
    source_entities: list[SourceEntity] = []

    if isinstance(parsed, ProductDetailsArguments):
        products: list[Product] = await get_products(
            session,
            as_of,
            model=parsed.model,
            product_group=parsed.product_group,
        )
        source_entities = products
        records: list[ToolRecord] = [_record_from_product(item) for item in products]
    elif isinstance(parsed, QuotePriceArguments):
        products: list[Product] = await get_products(
            session,
            as_of,
            model=parsed.model,
        )
        product: Product | None = products[0] if products else None
        regions: list[str] = []
        region_entities: list[Region] = []
        if parsed.province is not None:
            region_entities = await get_regions(
                session,
                as_of,
                province=parsed.province,
            )
            regions = list(
                dict.fromkeys(item.region for item in region_entities)
            )
            if parsed.region is not None:
                regions = [item for item in regions if item == parsed.region]
        elif parsed.region is not None:
            regions = [parsed.region]
        prices: list[Price] = []
        if parsed.province is None or regions:
            if regions:
                for region_name in regions:
                    prices.extend(
                        await get_prices(
                            session,
                            as_of,
                            model=parsed.model,
                            region=region_name,
                        )
                    )
            else:
                prices = await get_prices(session, as_of, model=parsed.model)
        promotions: list[Promotion] = await get_promotions_for_model(
            session,
            parsed.model,
        )
        source_entities = [
            *products,
            *prices,
            *promotions,
            *region_entities,
        ]
        if prices:
            records = [
                _quote_price_record(
                    model=parsed.model,
                    price=item,
                    product=product,
                    region=item.region,
                    promotions=promotions,
                    as_of=as_of,
                    channel=parsed.channel,
                )
                for item in prices
            ]
        elif product is not None:
            records = [
                _quote_price_record(
                    model=parsed.model,
                    price=None,
                    product=product,
                    region=regions[0] if regions else None,
                    promotions=promotions,
                    as_of=as_of,
                    channel=parsed.channel,
                )
            ]
        else:
            records = []
    elif isinstance(parsed, PromotionArguments):
        promotions: list[Promotion] = await get_promotions(
            session,
            as_of,
            model=parsed.model,
            channel=parsed.channel,
        )
        source_entities = promotions
        records = [
            _record_from_promotion(item)
            for item in promotions
        ]
    elif isinstance(parsed, WarrantyArguments):
        groups: list[str] = []
        if parsed.product_group is not None:
            groups = [parsed.product_group]
        elif parsed.model is not None:
            products = await get_products(session, as_of, model=parsed.model)
            groups = list(dict.fromkeys(item.product_group for item in products))
        warranties: list[Warranty] = []
        if parsed.product_group is not None or parsed.model is not None:
            for group in groups:
                warranties.extend(await get_warranty(session, as_of, group))
        else:
            warranties = await get_warranty(session, as_of)
        source_entities = warranties
        records = [
            _record_from_warranty(item)
            for item in warranties
        ]
    elif isinstance(parsed, ServiceCentreArguments):
        centres: list[ServiceCentre] = await find_service_centres(
            session,
            as_of,
            province=parsed.province,
            product_group=parsed.product_group,
            district=parsed.district,
        )
        source_entities = centres
        records = [_record_from_centre(item) for item in centres]
    elif isinstance(parsed, ServiceFeeArguments):
        fees: list[ServiceFee] = await get_service_fees(
            session,
            as_of,
            service_code=parsed.service_code,
            product_group=parsed.product_group,
            service_query=parsed.service_query,
        )
        source_entities = fees
        records = [_record_from_fee(item) for item in fees]
    elif isinstance(parsed, PolicySearchArguments):
        results: list[PDFSearchResult] = await retrieve_policy_chunks(
            qdrant_client=context.qdrant_client,
            embedding_model=context.embedding_model,
            question=parsed.query,
            as_of=as_of,
            base_collection_name=context.settings.qdrant_collection,
            embedding_model_name=context.settings.pdf_embedding_model,
        )
        records = [
            PolicyRecord(
                source_file=item.source_file,
                section_id=item.section_id,
                parent_section_id=item.parent_section_id,
                section_title=item.section_title,
                page=item.page,
                text=item.text,
                section_cite_ids=item.section_cite_ids,
                document_version=item.document_version,
                effective_date=item.effective_date,
                cite_id=item.cite_id,
            )
            for item in results
        ]
    elif isinstance(parsed, CalculateArguments):
        records = [
            CalculationRecord(
                expression=parsed.expression,
                result=_calculate_expression(parsed.expression),
            )
        ]
    elif isinstance(parsed, AddDaysArguments):
        weekday_names: tuple[str, ...] = (
            "Thứ Hai",
            "Thứ Ba",
            "Thứ Tư",
            "Thứ Năm",
            "Thứ Sáu",
            "Thứ Bảy",
            "Chủ Nhật",
        )
        result_date: date = parsed.start_date + timedelta(days=parsed.days)
        records = [
            AddDaysRecord(
                start_date=parsed.start_date,
                days=parsed.days,
                result_date=result_date,
                weekday=weekday_names[result_date.weekday()],
            )
        ]
    else:
        raise UnknownModelToolError(f"Tool arguments are not handled: {name}")

    source_notes: list[tuple[str, str]] = await _queried_worksheet_notes(
        session,
        name,
    )
    return _make_result(name, records, source_entities, source_notes)
