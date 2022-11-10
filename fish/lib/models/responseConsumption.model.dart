import 'package:json_annotation/json_annotation.dart';

part 'responseConsumption.model.g.dart';

@JsonSerializable()
class ResponseConsumption {
  String foodType;
  double foodAmount;
 

  ResponseConsumption({required this.foodType,required this.foodAmount});

  factory ResponseConsumption.fromJson(Map<String, dynamic> json) =>
      _$ResponseConsumptionFromJson(json);

  Map<String, dynamic> toJson() => _$ResponseConsumptionToJson(this);
}
