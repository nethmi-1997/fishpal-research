import 'package:json_annotation/json_annotation.dart';

part 'requestConsumption.model.g.dart';

@JsonSerializable()
class RequestConsumption {
  int species;
  int fishCount;
  int fishStage;
 

  RequestConsumption({required this.species,required this.fishCount,required this.fishStage});

  factory RequestConsumption.fromJson(Map<String, dynamic> json) =>
      _$RequestConsumptionFromJson(json);

  Map<String, dynamic> toJson() => _$RequestConsumptionToJson(this);
}
