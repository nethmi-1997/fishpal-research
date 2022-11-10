// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'responseConsumption.model.dart';

// **************************************************************************
// JsonSerializableGenerator
// **************************************************************************

ResponseConsumption _$ResponseConsumptionFromJson(Map<String, dynamic> json) =>
    ResponseConsumption(
      foodType: json['foodType'] as String,
      foodAmount: (json['foodAmount'] as num).toDouble(),
    );

Map<String, dynamic> _$ResponseConsumptionToJson(
        ResponseConsumption instance) =>
    <String, dynamic>{
      'foodType': instance.foodType,
      'foodAmount': instance.foodAmount,
    };
