from fastapi import APIRouter, Depends

from back.schemas.parse import ParseRequest, ParseResponse
from back.services.parser_service import ParserService


router = APIRouter(prefix="/api", tags=["parse"])


def get_parser_service() -> ParserService:
    from back.main import parser_service

    return parser_service


@router.post("/parse", response_model=ParseResponse)
def parse_sentence(
    request: ParseRequest,
    service: ParserService = Depends(get_parser_service),
) -> ParseResponse:
    return service.parse(request.text, request.maxTrees)

