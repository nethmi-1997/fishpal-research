import 'package:json_annotation/json_annotation.dart';

part 'responseBehaviour.model.g.dart';

@JsonSerializable()
class ResponseBehaviour {
  String type;
 

  ResponseBehaviour({required this.type});

  factory ResponseBehaviour.fromJson(Map<String, dynamic> json) =>
      _$ResponseBehaviourFromJson(json);

  Map<String, dynamic> toJson() => _$ResponseBehaviourToJson(this);
}
