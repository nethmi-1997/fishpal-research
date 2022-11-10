import 'package:json_annotation/json_annotation.dart';

part 'responseFilter.model.g.dart';

@JsonSerializable()
class ResponseFilter {
  String color;
 

  ResponseFilter({required this.color});

  factory ResponseFilter.fromJson(Map<String, dynamic> json) =>
      _$ResponseFilterFromJson(json);

  Map<String, dynamic> toJson() => _$ResponseFilterToJson(this);
}
