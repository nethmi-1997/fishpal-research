import 'package:json_annotation/json_annotation.dart';

part 'responseObjecDetection.model.g.dart';

@JsonSerializable()
class ResponseObjectDetection {
  double objectRes;
 

  ResponseObjectDetection({required this.objectRes});

  factory ResponseObjectDetection.fromJson(Map<String, dynamic> json) =>
      _$ResponseObjectDetectionFromJson(json);

  Map<String, dynamic> toJson() => _$ResponseObjectDetectionToJson(this);
}
