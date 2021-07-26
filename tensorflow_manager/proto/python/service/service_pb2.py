# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: service/service.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='service/service.proto',
  package='service',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x15service/service.proto\x12\x07service\"\x18\n\x07Request\x12\r\n\x05ready\x18\x01 \x01(\x08\"{\n\x0bHyperParams\x12/\n\x04loss\x18\x01 \x01(\x0e\x32!.service.HyperParams.LossFunction\x12\x12\n\nfromLogits\x18\x02 \x01(\x08\"\'\n\x0cLossFunction\x12\x17\n\x13SparseCategoricalCE\x10\x00\"[\n\rModelResponse\x12\r\n\x05model\x18\x01 \x01(\x0c\x12)\n\x0bhyperparams\x18\x02 \x01(\x0b\x32\x14.service.HyperParams\x12\x10\n\x08\x66ileName\x18\x03 \x01(\t2L\n\x0bModelEncode\x12=\n\x0fGetEncodedModel\x12\x10.service.Request\x1a\x16.service.ModelResponse\"\x00\x62\x06proto3'
)



_HYPERPARAMS_LOSSFUNCTION = _descriptor.EnumDescriptor(
  name='LossFunction',
  full_name='service.HyperParams.LossFunction',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='SparseCategoricalCE', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=144,
  serialized_end=183,
)
_sym_db.RegisterEnumDescriptor(_HYPERPARAMS_LOSSFUNCTION)


_REQUEST = _descriptor.Descriptor(
  name='Request',
  full_name='service.Request',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='ready', full_name='service.Request.ready', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=34,
  serialized_end=58,
)


_HYPERPARAMS = _descriptor.Descriptor(
  name='HyperParams',
  full_name='service.HyperParams',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='loss', full_name='service.HyperParams.loss', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='fromLogits', full_name='service.HyperParams.fromLogits', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _HYPERPARAMS_LOSSFUNCTION,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=60,
  serialized_end=183,
)


_MODELRESPONSE = _descriptor.Descriptor(
  name='ModelResponse',
  full_name='service.ModelResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='model', full_name='service.ModelResponse.model', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=b"",
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='hyperparams', full_name='service.ModelResponse.hyperparams', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='fileName', full_name='service.ModelResponse.fileName', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=185,
  serialized_end=276,
)

_HYPERPARAMS.fields_by_name['loss'].enum_type = _HYPERPARAMS_LOSSFUNCTION
_HYPERPARAMS_LOSSFUNCTION.containing_type = _HYPERPARAMS
_MODELRESPONSE.fields_by_name['hyperparams'].message_type = _HYPERPARAMS
DESCRIPTOR.message_types_by_name['Request'] = _REQUEST
DESCRIPTOR.message_types_by_name['HyperParams'] = _HYPERPARAMS
DESCRIPTOR.message_types_by_name['ModelResponse'] = _MODELRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Request = _reflection.GeneratedProtocolMessageType('Request', (_message.Message,), {
  'DESCRIPTOR' : _REQUEST,
  '__module__' : 'service.service_pb2'
  # @@protoc_insertion_point(class_scope:service.Request)
  })
_sym_db.RegisterMessage(Request)

HyperParams = _reflection.GeneratedProtocolMessageType('HyperParams', (_message.Message,), {
  'DESCRIPTOR' : _HYPERPARAMS,
  '__module__' : 'service.service_pb2'
  # @@protoc_insertion_point(class_scope:service.HyperParams)
  })
_sym_db.RegisterMessage(HyperParams)

ModelResponse = _reflection.GeneratedProtocolMessageType('ModelResponse', (_message.Message,), {
  'DESCRIPTOR' : _MODELRESPONSE,
  '__module__' : 'service.service_pb2'
  # @@protoc_insertion_point(class_scope:service.ModelResponse)
  })
_sym_db.RegisterMessage(ModelResponse)



_MODELENCODE = _descriptor.ServiceDescriptor(
  name='ModelEncode',
  full_name='service.ModelEncode',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=278,
  serialized_end=354,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetEncodedModel',
    full_name='service.ModelEncode.GetEncodedModel',
    index=0,
    containing_service=None,
    input_type=_REQUEST,
    output_type=_MODELRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_MODELENCODE)

DESCRIPTOR.services_by_name['ModelEncode'] = _MODELENCODE

# @@protoc_insertion_point(module_scope)
