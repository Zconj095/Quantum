using UnityEngine;

public class PhononPhotonEMInteraction : MonoBehaviour
{
    public float phononWeight = 0.7f; // Higher weight for phonons
    public float photonWeight = 0.3f; // Lower weight for photons
    public Vector3 emFieldStrength = new Vector3(0, 1, 0); // Represents the EM field strength directionally

    private float phononVelocityChange = 0f;
    private float photonVelocityChange = 0f;
    private float subjectiveRateOfChange = 0f;

    void Start()
    {
        // Initialize with default velocity changes
        // In a real scenario, these values could be dynamically calculated based on the EM field
        phononVelocityChange = CalculateVelocityChange(true, emFieldStrength);
        photonVelocityChange = CalculateVelocityChange(false, emFieldStrength);

        // Calculate the initial subjective rate of change
        subjectiveRateOfChange = CalculateSubjectiveRateOfChange();
        Debug.Log("Initial Subjective Rate of Change: " + subjectiveRateOfChange);
    }

    void Update()
    {
        // Example of dynamically updating EM field strength and its effects
        // Here we're just simulating this by randomly adjusting the EM field strength
        emFieldStrength = new Vector3(Random.Range(-1f, 1f), Random.Range(-1f, 1f), Random.Range(-1f, 1f));

        // Recalculate velocity changes based on the updated EM field
        phononVelocityChange = CalculateVelocityChange(true, emFieldStrength);
        photonVelocityChange = CalculateVelocityChange(false, emFieldStrength);

        // Recalculate the subjective rate of change
        subjectiveRateOfChange = CalculateSubjectiveRateOfChange();
        Debug.Log("Updated Subjective Rate of Change: " + subjectiveRateOfChange);
    }

    float CalculateVelocityChange(bool isPhonon, Vector3 emField)
    {
        // Simplified function to calculate velocity change based on EM field strength
        // This is a placeholder for a more complex calculation involving material properties
        return isPhonon ? emField.magnitude * 0.1f : emField.magnitude * 0.05f;
    }

    float CalculateSubjectiveRateOfChange()
    {
        // Calculate the subjective rate of change based on weighted phonon and photon velocity changes
        return (phononWeight * phononVelocityChange) + (photonWeight * photonVelocityChange);
    }
}
