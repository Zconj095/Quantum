using UnityEngine;

public class QuantumSimulation : MonoBehaviour
{
    public GameObject photonPrefab; // Assign a light particle prefab
    public GameObject phononPrefab; // Assign a sound particle prefab

    private GameObject photon;
    private GameObject phonon;

    void Start()
    {
        // Corrected lines with explicit casting
        photon = Instantiate(photonPrefab, new Vector3(-5, 0, 0), Quaternion.identity) as GameObject;
        phonon = Instantiate(phononPrefab, new Vector3(5, 2, 0), Quaternion.identity) as GameObject;
    }

    void Update()
    {
        // Move photon at a constant speed to simulate light
        photon.transform.Translate(Vector3.right * Time.deltaTime * 5);

        // Oscillate phonon to simulate sound vibration
        phonon.transform.position = new Vector3(5, 2 + Mathf.Sin(Time.time * 5), 0);
    }
}
