from ilastik.applets.base.appletSerializer import AppletSerializer, SerialSlot


class Groups(object):
    Labels = "Labels"


class ObjectClassificationSerializer(AppletSerializer):
    """
    """
    def __init__(self, topGroupName, operator):
        serialSlots = []
        super(ObjectClassificationSerializer, self ).__init__(topGroupName,
                                                              slots=serialSlots,
                                                              operator=operator)
