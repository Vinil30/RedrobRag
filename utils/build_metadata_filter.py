from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
)


class MetadataFilter:

    def __init__(self):
        pass

    def filter_builder(self, hard_rejects):

        must = []
        must_not = []

        for hr in hard_rejects:
            if hr.operator == "==":
                must.append(
                    FieldCondition(
                        key=hr.lhs,
                        match=MatchValue(value=hr.rhs)
                    )
                )

            elif hr.operator == "!=":
                must_not.append(
                    FieldCondition(
                        key=hr.lhs,
                        match=MatchValue(value=hr.rhs)
                    )
                )

            elif hr.operator == "in":
                must.append(
                    FieldCondition(
                        key=hr.lhs,
                        match=MatchAny(any=hr.rhs)
                    )
                )

            elif hr.operator == "not_in":
                must_not.append(
                    FieldCondition(
                        key=hr.lhs,
                        match=MatchAny(any=hr.rhs)
                    )
                )

            elif hr.operator == ">=":
                must.append(
                    FieldCondition(
                        key=hr.lhs,
                        range=Range(gte=hr.rhs)
                    )
                )

            elif hr.operator == ">":
                must.append(
                    FieldCondition(
                        key=hr.lhs,
                        range=Range(gt=hr.rhs)
                    )
                )

            elif hr.operator == "<=":
                must.append(
                    FieldCondition(
                        key=hr.lhs,
                        range=Range(lte=hr.rhs)
                    )
                )

            elif hr.operator == "<":
                must.append(
                    FieldCondition(
                        key=hr.lhs,
                        range=Range(lt=hr.rhs)
                    )
                )

        return Filter(
            must=must,
            must_not=must_not
        )