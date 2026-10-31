"""Typed, provider-neutral tools for an LLM or agent to query source materials."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, DecimalException
from typing import Annotated, Literal, TypeAlias
from urllib.parse import quote

from fastembed import TextEmbedding
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

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
    get_products,
    get_promotions,
    get_regions,
    get_prices,
    get_service_fees,
    get_warranty,
)
from app.tools.pdf_retriever import PDFSearchResult, retrieve_policy_chunks


class ToolArguments(BaseModel):
    """Provide shared time context and strict argument validation for tools.

    Attributes:
        as_of: Timestamp of the customer's question, used for effective-date
            filtering.

    Raises:
        ValueError: If any declared argument is invalid.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

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


class RegionArguments(ToolArguments):
    """Describe a province-to-price-region lookup.

    Attributes:
        province: Optional exact province or city.
    """

    province: str | None = Field(
        default=None,
        description="Tỉnh/thành cần tra vùng giá.",
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


class DateContextArguments(ToolArguments):
    """Request calendar context for the question's timestamp.

    Attributes:
        as_of: Timestamp of the customer's question.
    """


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
        region: Price region.
        price: Effective listed price.
        valid_from: Date the listed price became effective.
        cite_id: Workbook, sheet, and source row citation.

    Raises:
        ValidationError: If returned values do not match the record schema.
    """

    kind: Literal["price"] = "price"
    model: str
    region: str
    price: Decimal
    valid_from: date


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


class RegionRecord(CitedToolRecord):
    """Represent the price-region mapping for a province.

    Attributes:
        kind: Region record discriminator.
        province: Province or city from the source.
        region: Mapped price region.
        cite_id: Workbook, sheet, and source row citation.

    Raises:
        ValidationError: If returned values do not match the record schema.
    """

    kind: Literal["region"] = "region"
    province: str
    region: str


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


class DateContextRecord(BaseModel):
    """Represent the request timestamp and its Vietnamese calendar context.

    Attributes:
        kind: Identifies this record as date context.
        as_of: Original request timestamp.
        local_date: Calendar date in the timestamp's own timezone.
        weekday: Vietnamese weekday name.
    """

    kind: Literal["date_context"] = "date_context"
    as_of: datetime
    local_date: str
    weekday: str


ToolRecord: TypeAlias = Annotated[
    ProductRecord
    | PriceRecord
    | PromotionRecord
    | WarrantyRecord
    | RegionRecord
    | ServiceCentreRecord
    | ServiceFeeRecord
    | PolicyRecord
    | CalculationRecord
    | DateContextRecord,
    Field(discriminator="kind"),
]


class WorksheetNoteRecord(BaseModel):
    """Represent a worksheet note or footer row with its original row citation.

    Attributes:
        source_row: Original one-based Excel row.
        values: Non-empty cells keyed by Excel column letters.
        cite_id: Citation for the workbook, sheet, and note row.
    """

    source_row: int
    values: dict[str, str | int | float | bool | None]
    cite_id: str


class WorksheetSourceContext(BaseModel):
    """Provide table headings and surrounding notes for worksheet results.

    Attributes:
        source_file: Relative source workbook path.
        sheet_name: Original worksheet title.
        cite_id: Sheet-level reference for the configured table headings.
        columns: Flattened headers, preserving multi-row title hierarchy.
        header_notes: Non-tabular content before the configured table headers.
        footnotes: Non-tabular content after the table data.
    """

    source_file: str
    sheet_name: str
    cite_id: str
    columns: list[str]
    header_notes: list[WorksheetNoteRecord]
    footnotes: list[WorksheetNoteRecord]


class ToolResult(BaseModel):
    """Return typed records and the exact citations available to the model.

    Attributes:
        tool_name: Name of the tool that was executed.
        records: Typed results from the corresponding source.
        citations: Unique citations attached to returned source records.
        source_contexts: Worksheet headers and surrounding notes for source rows.
    """

    tool_name: str
    records: list[ToolRecord]
    citations: list[str]
    source_contexts: list[WorksheetSourceContext] = Field(default_factory=list)


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
        "Tra cứu thông số, nhóm và trạng thái sản phẩm trong danh mục.",
    ),
    "quote_price": (
        QuotePriceArguments,
        "Tra cứu giá model theo vùng hoặc tự xác định vùng từ tỉnh/thành.",
    ),
    "get_promotions": (
        PromotionArguments,
        "Tra cứu khuyến mãi đang có hiệu lực theo ngày hỏi, model và kênh.",
    ),
    "get_warranty": (
        WarrantyArguments,
        "Tra cứu thời hạn và điều kiện bảo hành theo model hoặc nhóm sản phẩm.",
    ),
    "get_region": (
        RegionArguments,
        "Tra cứu vùng giá tương ứng với tỉnh/thành.",
    ),
    "find_service_centres": (
        ServiceCentreArguments,
        "Tìm trung tâm dịch vụ theo địa điểm và nhóm sản phẩm được tiếp nhận.",
    ),
    "get_service_fee": (
        ServiceFeeArguments,
        "Tra cứu phí dịch vụ trong và ngoài bảo hành.",
    ),
    "search_policy": (
        PolicySearchArguments,
        "Tìm điều khoản liên quan trong sổ tay chính sách; trả về trích dẫn.",
    ),
    "calculate": (
        CalculateArguments,
        "Tính biểu thức số học bằng số liệu đã lấy từ các công cụ khác.",
    ),
    "get_date_context": (
        DateContextArguments,
        "Lấy ngày và thứ trong tuần của thời điểm câu hỏi.",
    ),
}


class UnknownModelToolError(ValueError):
    """Indicate that a requested model tool is not registered."""


def get_model_tool_definitions() -> list[ModelToolDefinition]:
    """Return model-facing JSON-schema function declarations.

    Returns:
        list[ModelToolDefinition]: Tool names, descriptions, and input schemas.
    """
    return [
        ModelToolDefinition(
            function=FunctionDefinition(
                name=name,
                description=description,
                parameters=arguments_model.model_json_schema(),
            )
        )
        for name, (arguments_model, description) in TOOL_DESCRIPTIONS.items()
    ]


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


def _record_from_price(price: Price) -> PriceRecord:
    """Convert a typed price row into a model-facing result record.

    Args:
        price: Effective price row with eagerly-loaded source provenance.

    Returns:
        PriceRecord: Price, region, effective date, and citation.
    """
    return PriceRecord(
        model=price.model,
        region=price.region,
        price=price.price,
        valid_from=price.valid_from,
        cite_id=price.cite_id,
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


def _record_from_region(region: Region) -> RegionRecord:
    """Convert a province mapping row into a model-facing result record.

    Args:
        region: Region row with eagerly-loaded source provenance.

    Returns:
        RegionRecord: Province mapping and citation.
    """
    return RegionRecord(
        province=region.province,
        region=region.region,
        cite_id=region.cite_id,
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


def _note_records(
    source_file: str,
    sheet_name: str,
    notes: list[dict[str, object]],
) -> list[WorksheetNoteRecord]:
    """Convert persisted note rows into typed records with row citations.

    Args:
        source_file: Relative workbook path.
        sheet_name: Original worksheet title.
        notes: Raw note objects stored during workbook ingestion.

    Returns:
        list[WorksheetNoteRecord]: Notes retaining their workbook row numbers.
    """
    encoded_sheet: str = quote(sheet_name, safe="")
    result: list[WorksheetNoteRecord] = []
    for note in notes:
        source_row: object = note.get("source_row")
        values: object = note.get("values")
        if not isinstance(source_row, int) or not isinstance(values, dict):
            raise ValueError(
                f"Invalid persisted worksheet note for {source_file} "
                f"sheet {sheet_name!r}."
            )
        result.append(
            WorksheetNoteRecord(
                source_row=source_row,
                values=values,
                cite_id=(
                    f"{source_file}#sheet={encoded_sheet}&row={source_row}"
                ),
            )
        )
    return result


def _worksheet_contexts(
    source_entities: list[SourceEntity],
) -> list[WorksheetSourceContext]:
    """Collect unique header and footer context for source worksheet rows.

    Args:
        source_entities: Normalized source rows with eager-loaded provenance.

    Returns:
        list[WorksheetSourceContext]: One full worksheet context per source
            sheet returned by the tool.
    """
    contexts: dict[int, WorksheetSourceContext] = {}
    for entity in source_entities:
        sheet: SourceSheet = entity.source_row.sheet
        if sheet.id in contexts:
            continue
        source_file: str = sheet.workbook.source_file
        encoded_sheet: str = quote(sheet.name, safe="")
        contexts[sheet.id] = WorksheetSourceContext(
            source_file=source_file,
            sheet_name=sheet.name,
            cite_id=f"{source_file}#sheet={encoded_sheet}",
            columns=sheet.columns or [],
            header_notes=_note_records(
                source_file,
                sheet.name,
                sheet.header_notes or [],
            ),
            footnotes=_note_records(
                source_file,
                sheet.name,
                sheet.footnotes or [],
            ),
        )
    return list(contexts.values())


def _make_result(
    name: str,
    records: list[ToolRecord],
    source_entities: list[SourceEntity] | None = None,
) -> ToolResult:
    """Build a tool response with row and worksheet-context citations.

    Args:
        name: Registered tool name.
        records: Typed result records.
        source_entities: Optional normalized rows used to resolve worksheets.

    Returns:
        ToolResult: Records and citations limited to this invocation.
    """
    citation_values: list[str] = []
    for record in records:
        if isinstance(record, PolicyRecord):
            citation_values.extend(record.section_cite_ids)
        elif isinstance(record, CitedToolRecord):
            citation_values.append(record.cite_id)
    source_contexts: list[WorksheetSourceContext] = _worksheet_contexts(
        source_entities or []
    )
    for source_context in source_contexts:
        citation_values.append(source_context.cite_id)
        citation_values.extend(
            note.cite_id
            for note in [*source_context.header_notes, *source_context.footnotes]
        )
    citations: list[str] = list(dict.fromkeys(citation_values))
    return ToolResult(
        tool_name=name,
        records=records,
        citations=citations,
        source_contexts=source_contexts,
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
        regions: list[str] = []
        if parsed.province is not None:
            mappings: list[Region] = await get_regions(
                session,
                as_of,
                province=parsed.province,
            )
            regions = list(dict.fromkeys(item.region for item in mappings))
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
        source_entities = prices
        records = [_record_from_price(item) for item in prices]
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
    elif isinstance(parsed, RegionArguments):
        mappings = await get_regions(session, as_of, province=parsed.province)
        source_entities = mappings
        records = [_record_from_region(item) for item in mappings]
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
    elif isinstance(parsed, DateContextArguments):
        weekday_names: tuple[str, ...] = (
            "Thứ Hai",
            "Thứ Ba",
            "Thứ Tư",
            "Thứ Năm",
            "Thứ Sáu",
            "Thứ Bảy",
            "Chủ Nhật",
        )
        records = [
            DateContextRecord(
                as_of=as_of,
                local_date=as_of.date().isoformat(),
                weekday=weekday_names[as_of.weekday()],
            )
        ]
    else:
        raise UnknownModelToolError(f"Tool arguments are not handled: {name}")

    return _make_result(name, records, source_entities)
